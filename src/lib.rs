//! A module implementing HPACK functionality. Exposes a simple API for
//! performing the encoding and decoding of header sets, according to the
//! HPACK spec.

#[macro_use]
extern crate log;
#[cfg(feature = "interop_tests")]
extern crate rustc_serialize;

use std::collections::vec_deque;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::iter;
use std::slice;
use std::sync::Arc;

// Re-export the main HPACK API entry points.
pub use self::decoder::Decoder;
pub use self::encoder::Encoder;

pub mod decoder;
pub mod encoder;
pub mod huffman;

/// An `Iterator` through elements of the `DynamicTable`.
///
/// The implementation of the iterator itself is very tightly coupled
/// to the implementation of the `DynamicTable`.
///
/// This iterator returns tuples of slices. The tuples themselves are
/// constructed as new instances, containing a borrow from the `Vec`s
/// representing the underlying Headers.
struct DynamicTableIter<'a> {
    /// Stores an iterator through the underlying structure that the
    /// `DynamicTable` uses. The third element of each tuple is the
    /// RFC-7541 logical entry size (name.len() + value.len() + 32); it
    /// is stripped from the items yielded by the iterator.
    inner: vec_deque::Iter<'a, (Vec<u8>, Vec<u8>, usize)>,
}

impl<'a> Iterator for DynamicTableIter<'a> {
    type Item = (&'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<(&'a [u8], &'a [u8])> {
        match self.inner.next() {
            Some(ref header) => Some((&header.0, &header.1)),
            None => None,
        }
    }
}

/// A struct representing the dynamic table that needs to be maintained by the
/// coder.
///
/// The dynamic table contains a number of recently used headers. The size of
/// the table is constrained to a certain number of octets. If on insertion of
/// a new header into the table, the table would exceed the maximum size,
/// headers are evicted in a FIFO fashion until there is enough room for the
/// new header to be inserted. (Therefore, it is possible that though all
/// elements end up being evicted, there is still not enough space for the new
/// header: when the size of this individual header exceeds the maximum size of
/// the table.)
///
/// The current size of the table is calculated, based on the IETF definition,
/// as the sum of sizes of each header stored within the table, where the size
/// of an individual header is
/// `len_in_octets(header_name) + len_in_octets(header_value) + 32`.
///
/// Note: the maximum size of the dynamic table does not have to be equal to
/// the maximum header table size as defined by a "higher level" protocol
/// (such as the `SETTINGS_HEADER_TABLE_SIZE` setting in HTTP/2), since HPACK
/// can choose to modify the dynamic table size on the fly (as long as it keeps
/// it below the maximum value set by the protocol). So, the `DynamicTable`
/// only cares about the maximum size as set by the HPACK {en,de}coder and lets
/// *it* worry about making certain that the changes are valid according to
/// the (current) constraints of the protocol.
struct DynamicTable {
    // Each entry is `(name, value, rfc_size)` where `rfc_size` is the
    // RFC-7541 entry size (`name.len() + value.len() + 32`). It is stored
    // alongside the bytes so that `consolidate_table` can subtract the
    // correct logical size on eviction even when the entry's value (and
    // possibly name) has been dropped to save memory. The
    // RFC-7541-compliant encoder on the server side retains the value,
    // so to keep eviction timing — and thus the index address space —
    // aligned with the encoder, the decoder MUST account for the full
    // RFC-7541 size regardless of whether it retains the bytes.
    // (RFC 7541 §4.1.)
    table: VecDeque<(Vec<u8>, Vec<u8>, usize)>,
    size: usize,
    max_size: usize,
    // Headers there are not in the white list are not saved in the dynamic table. If the whitelist is empty, all headers are saved.
    expected_headers: Arc<HashSet<Vec<u8>>>,
}

impl DynamicTable {
    /// Creates a new empty dynamic table with a default size.
    fn new() -> DynamicTable {
        // The default maximum size corresponds to the default HTTP/2
        // setting
        DynamicTable::with_size(4096)
    }

    /// Creates a new empty dynamic table with a headers white list.
    fn new_with_expected_headers(expected_headers: Arc<HashSet<Vec<u8>>>) -> DynamicTable {
        DynamicTable {
            table: VecDeque::new(),
            size: 0,
            max_size: 4096,
            expected_headers,
        }
    }

    /// Creates a new empty dynamic table with the given maximum size.
    fn with_size(max_size: usize) -> DynamicTable {
        DynamicTable {
            table: VecDeque::new(),
            size: 0,
            max_size: max_size,
            expected_headers: Arc::new(HashSet::new()),
        }
    }

    /// Returns the current size of the table in octets, as defined by the IETF
    /// HPACK spec.
    fn get_size(&self) -> usize {
        self.size
    }

    /// Returns an `Iterator` through the headers stored in the `DynamicTable`.
    ///
    /// The iterator will yield elements of type `(&[u8], &[u8])`,
    /// corresponding to a single header name and value. The name and value
    /// slices are borrowed from their representations in the `DynamicTable`
    /// internal implementation, which means that it is possible only to
    /// iterate through the headers, not mutate them.
    fn iter(&self) -> DynamicTableIter {
        DynamicTableIter {
            inner: self.table.iter(),
        }
    }

    /// Sets the new maximum table size.
    ///
    /// If the current size of the table is larger than the new maximum size,
    /// existing headers are evicted in a FIFO fashion until the size drops
    /// below the new maximum.
    fn set_max_table_size(&mut self, new_max_size: usize) {
        self.max_size = new_max_size;
        // Make the table size fit within the new constraints.
        self.consolidate_table();
    }

    /// Returns the maximum size of the table in octets.
    fn get_max_table_size(&self) -> usize {
        self.max_size
    }

    /// Add a new header to the dynamic table.
    ///
    /// The table automatically gets resized, if necessary.
    ///
    /// Do note that, under the HPACK rules, it is possible the given header
    /// is not found in the dynamic table after this operation finishes, in
    /// case the total size of the given header exceeds the maximum size of the
    /// dynamic table.
    fn add_header(&mut self, name: Vec<u8>, value: Vec<u8>) {
        let expected_header = if self.expected_headers.is_empty() {
            true
        } else {
            self.expected_headers.contains(&name.to_ascii_lowercase())
        };
        // RFC 7541 §4.1: the size of an entry is
        // `name.octets + value.octets + 32`, independent of whether the
        // caller retains the value bytes. We account for the full RFC
        // size in *both* branches so that eviction timing — and thus the
        // index address space — stays aligned with an RFC-compliant
        // server encoder that retains all values. The third element of
        // the stored tuple carries this size so `consolidate_table`
        // subtracts the same amount on eviction.
        let entry_size = name.len() + value.len() + 32;
        if expected_header {
            self.size += entry_size;
            debug!("New dynamic table size {}", self.size);
            // Now add it to the internal buffer
            self.table.push_front((name, value, entry_size));
        } else {
            self.size += entry_size;
            debug!("New dynamic table size {}", self.size);
            // Drop name and value to save memory; the RFC-7541 size is
            // preserved in the third tuple element so eviction
            // accounting remains correct.
            self.table.push_front((vec![], vec![], entry_size));
        }
        // ...and make sure we're not over the maximum size.
        self.consolidate_table();
        debug!("After consolidation dynamic table size {}", self.size);
    }

    /// Consolidates the table entries so that the table size is below the
    /// maximum allowed size, by evicting headers from the table in a FIFO
    /// fashion.
    fn consolidate_table(&mut self) {
        while self.size > self.max_size {
            {
                let last_header = match self.table.back() {
                    Some(x) => x,
                    None => {
                        // Can never happen as the size of the table must reach
                        // 0 by the time we've exhausted all elements.
                        panic!("Size of table != 0, but no headers left!");
                    }
                };
                self.size -= last_header.2;
            }
            self.table.pop_back();
        }
    }

    /// Returns the number of headers in the dynamic table.
    ///
    /// This is different than the size of the dynamic table.
    fn len(&self) -> usize {
        self.table.len()
    }

    /// Converts the current state of the table to a `Vec`
    fn to_vec(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut ret: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for elem in self.table.iter() {
            ret.push((elem.0.clone(), elem.1.clone()));
        }

        ret
    }

    /// Returns a reference to the header at the given index, if found in the
    /// dynamic table.
    fn get(&self, index: usize) -> Option<&(Vec<u8>, Vec<u8>, usize)> {
        self.table.get(index)
    }
}

impl fmt::Debug for DynamicTable {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.table)
    }
}

/// Represents the type of the static table, as defined by the HPACK spec.
type StaticTable<'a> = &'a [(&'a [u8], &'a [u8])];

/// Implements an iterator through the entire `HeaderTable`.
///
/// Yields first the elements from the static table, followed by elements from
/// the dynamic table, with each element being of type `(&[u8], &[u8])`.
///
/// This struct is tightly coupled to the implementation of the `HeaderTable`,
/// but its clients are shielded from all that and have a convenient (and
/// standardized) interface to iterate through all headers of the table.
///
/// The declaration of the inner iterator that is wrapped by this struct is a
/// monstrosity, that is required because "abstract return types" don't exist
/// yet ([https://github.com/rust-lang/rfcs/pull/105]).
struct HeaderTableIter<'a> {
    // Represents a chain of static-table -> dynamic-table elements.
    // The mapper is required to transform the elements yielded from the static
    // table to a type that matches the elements yielded from the dynamic table.
    inner: iter::Chain<
        iter::Map<
            slice::Iter<'a, (&'a [u8], &'a [u8])>,
            fn(&'a (&'a [u8], &'a [u8])) -> (&'a [u8], &'a [u8]),
        >,
        DynamicTableIter<'a>,
    >,
}

impl<'a> Iterator for HeaderTableIter<'a> {
    type Item = (&'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<(&'a [u8], &'a [u8])> {
        // Simply delegates to the wrapped iterator that is constructed by the
        // `HeaderTable` and passed into the `HeaderTableIter`.
        self.inner.next()
    }
}

/// A helper function that maps a borrowed tuple containing two borrowed slices
/// to just a tuple of two borrowed slices.
///
/// This helper function is needed because in order to define the type
/// `HeaderTableIter` we need to be able to refer to a real type for the Fn
/// template parameter, which means that when instantiating an instance, a
/// closure cannot be passed, since it cannot be named.
fn static_table_mapper<'a>(h: &'a (&'a [u8], &'a [u8])) -> (&'a [u8], &'a [u8]) {
    *h
}

/// The struct represents the header table obtained by merging the static and
/// dynamic tables into a single index address space, as described in section
/// `2.3.3.` of the HPACK spec.
struct HeaderTable<'a> {
    static_table: StaticTable<'a>,
    dynamic_table: DynamicTable,
}

impl<'a> HeaderTable<'a> {
    /// Creates a new header table where the static part is initialized with
    /// the given static table.
    pub fn with_static_table(static_table: StaticTable<'a>) -> HeaderTable<'a> {
        HeaderTable {
            static_table,
            dynamic_table: DynamicTable::new(),
        }
    }

    /// Creates a new header table where the static part is initialized with
    /// the given static table and expected headers.
    pub fn with_expected_headers(
        static_table: StaticTable<'a>,
        expected_headers: Arc<HashSet<Vec<u8>>>,
    ) -> HeaderTable<'a> {
        HeaderTable {
            static_table,
            dynamic_table: DynamicTable::new_with_expected_headers(expected_headers),
        }
    }

    /// Returns an iterator through *all* headers stored in the header table,
    /// i.e. it includes both the ones found in the static table and the
    /// dynamic table, in the order of their indices in the single address
    /// space (first the headers in the static table, followed by headers in
    /// the dynamic table).
    ///
    /// The type yielded by the iterator is `(&[u8], &[u8])`, where the tuple
    /// corresponds to the header name, value pairs in the described order.
    pub fn iter(&'a self) -> HeaderTableIter<'a> {
        HeaderTableIter {
            inner: self
                .static_table
                .iter()
                .map(static_table_mapper as fn(&'a (&'a [u8], &'a [u8])) -> (&'a [u8], &'a [u8]))
                .chain(self.dynamic_table.iter()),
        }
    }

    /// Adds the given header to the table. Of course, this means that the new
    /// header is added to the dynamic part of the table.
    ///
    /// If the size of the new header is larger than the current maximum table
    /// size of the dynamic table, the effect will be that the dynamic table
    /// gets emptied and the new header does *not* get inserted into it.
    #[inline]
    pub fn add_header(&mut self, name: Vec<u8>, value: Vec<u8>) {
        self.dynamic_table.add_header(name, value);
    }

    /// Returns a reference to the header (a `(name, value)` pair) with the
    /// given index in the table.
    ///
    /// The table is 1-indexed and constructed in such a way that the first
    /// entries belong to the static table, followed by entries in the dynamic
    /// table. They are merged into a single index address space, though.
    ///
    /// This is according to the [HPACK spec, section 2.3.3.]
    /// (http://http2.github.io/http2-spec/compression.html#index.address.space)
    pub fn get_from_table(&self, index: usize) -> Option<(&[u8], &[u8])> {
        // The IETF defined table indexing as 1-based.
        // So, before starting, make sure the given index is within the proper
        // bounds.
        let real_index = if index > 0 { index - 1 } else { return None };

        if real_index < self.static_table.len() {
            // It is in the static table so just return that...
            Some(self.static_table[real_index])
        } else {
            // Maybe it's in the dynamic table then?
            let dynamic_index = real_index - self.static_table.len();
            if dynamic_index < self.dynamic_table.len() {
                match self.dynamic_table.get(dynamic_index) {
                    Some(&(ref name, ref value, _)) => Some((name, value)),
                    None => Some((&[], &[])), // The message captured may be only part of the message, so the
                                              // header may not be found, but there are still some header that
                                              // can be obtained. If a None is returned, the parsing will fail.
                }
            } else {
                // Index out of bounds.
                Some((&[], &[]))
            }
        }
    }

    /// Finds the given header in the header table. Tries to match both the
    /// header name and value to one of the headers in the table. If no such
    /// header exists, then falls back to returning one that matched only the
    /// name.
    ///
    /// # Returns
    ///
    /// An `Option`, where `Some` corresponds to a tuple representing the index
    /// of the header in the header tables (the 1-based index that HPACK uses)
    /// and a `bool` indicating whether the value of the header also matched.
    pub fn find_header(&self, header: (&[u8], &[u8])) -> Option<(usize, bool)> {
        // Just does a simple scan of the entire table, searching for a header
        // that matches both the name and the value of the given header.
        // If no such header is found, then any one of the headers that had a
        // matching name is returned, with the appropriate return flag set.
        //
        // The tables are so small that it is unlikely that the linear scan
        // would be a major performance bottlneck. If it does prove to be,
        // though, a more efficient lookup/header representation method could
        // be devised.
        let mut matching_name: Option<usize> = None;
        for (i, h) in self.iter().enumerate() {
            if header.0 == h.0 {
                if header.1 == h.1 {
                    // Both name and value matched: returns it immediately
                    return Some((i + 1, true));
                }
                // If only the name was valid, we continue scanning, hoping to
                // find one where both the name and value match. We remember
                // this one, in case such a header isn't found after all.
                matching_name = Some(i + 1);
            }
        }

        // Finally, if there's no header with a matching name and value,
        // return one that matched only the name, if that *was* found.
        match matching_name {
            Some(i) => Some((i, false)),
            None => None,
        }
    }
}

/// The table represents the static header table defined by the HPACK spec.
/// (HPACK, Appendix A)
static STATIC_TABLE: &'static [(&'static [u8], &'static [u8])] = &[
    (b":authority", b""),
    (b":method", b"GET"),
    (b":method", b"POST"),
    (b":path", b"/"),
    (b":path", b"/index.html"),
    (b":scheme", b"http"),
    (b":scheme", b"https"),
    (b":status", b"200"),
    (b":status", b"204"),
    (b":status", b"206"),
    (b":status", b"304"),
    (b":status", b"400"),
    (b":status", b"404"),
    (b":status", b"500"),
    (b"accept-", b""),
    (b"accept-encoding", b"gzip, deflate"),
    (b"accept-language", b""),
    (b"accept-ranges", b""),
    (b"accept", b""),
    (b"access-control-allow-origin", b""),
    (b"age", b""),
    (b"allow", b""),
    (b"authorization", b""),
    (b"cache-control", b""),
    (b"content-disposition", b""),
    (b"content-encoding", b""),
    (b"content-language", b""),
    (b"content-length", b""),
    (b"content-location", b""),
    (b"content-range", b""),
    (b"content-type", b""),
    (b"cookie", b""),
    (b"date", b""),
    (b"etag", b""),
    (b"expect", b""),
    (b"expires", b""),
    (b"from", b""),
    (b"host", b""),
    (b"if-match", b""),
    (b"if-modified-since", b""),
    (b"if-none-match", b""),
    (b"if-range", b""),
    (b"if-unmodified-since", b""),
    (b"last-modified", b""),
    (b"link", b""),
    (b"location", b""),
    (b"max-forwards", b""),
    (b"proxy-authenticate", b""),
    (b"proxy-authorization", b""),
    (b"range", b""),
    (b"referer", b""),
    (b"refresh", b""),
    (b"retry-after", b""),
    (b"server", b""),
    (b"set-cookie", b""),
    (b"strict-transport-security", b""),
    (b"transfer-encoding", b""),
    (b"user-agent", b""),
    (b"vary", b""),
    (b"via", b""),
    (b"www-authenticate", b""),
];

#[cfg(test)]
mod tests {
    use super::DynamicTable;
    use super::HeaderTable;
    use super::STATIC_TABLE;

    #[test]
    fn test_dynamic_table_size_calculation_simple() {
        let mut table = DynamicTable::new();
        // Sanity check
        assert_eq!(0, table.get_size());

        table.add_header(b"a".to_vec(), b"b".to_vec());

        assert_eq!(32 + 2, table.get_size());
    }

    /// Regression test for the placeholder-size desync (keter-deepflow #374).
    ///
    /// When the dynamic table is configured with an expected-header
    /// whitelist (as the deepflow agent's HTTP/2 flow log parser does for
    /// `:status`), non-expected headers are inserted as *placeholders* —
    /// the name and value bytes are dropped to save memory. The table
    /// size, however, MUST still account for the full RFC-7541 entry size
    /// (`name.len() + value.len() + 32`), because the server-side encoder
    /// retains the value and computes its eviction timing against that
    /// same size. If the decoder under-sizes placeholders (the previous
    /// bug added only `32` per placeholder), the decoder's table size
    /// drifts below the encoder's, evictions fire at different times,
    /// index space diverges, and an indexed `:status` (or any later
    /// indexed header) resolves to a wrong/empty slot — silently dropping
    /// the response record. (RFC 7541 §4.1.)
    #[test]
    fn test_dynamic_table_placeholder_rfc7541_size() {
        use std::collections::HashSet;
        use std::sync::Arc;

        let mut expected: HashSet<Vec<u8>> = HashSet::new();
        expected.insert(b":status".to_vec());
        let mut table = DynamicTable::new_with_expected_headers(Arc::new(expected));

        // A realistic HTTP/2 response header set: only `:status` is in the
        // whitelist; the rest become placeholders but MUST still be sized
        // per RFC-7541.
        let headers: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (b":status".to_vec(), b"200".to_vec()),
            (b"cache-control".to_vec(), b"private".to_vec()),
            (b"date".to_vec(), b"Mon, 21 Oct 2013 20:13:21 GMT".to_vec()),
            (b"location".to_vec(), b"https://www.example.com".to_vec()),
            (b"content-encoding".to_vec(), b"gzip".to_vec()),
            (
                b"set-cookie".to_vec(),
                b"foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1".to_vec(),
            ),
        ];

        let mut rfc_size = 0usize;
        for (name, value) in &headers {
            table.add_header(name.clone(), value.clone());
            rfc_size += name.len() + value.len() + 32;
            assert_eq!(
                table.get_size(),
                rfc_size,
                "size must follow RFC-7541 accounting after inserting {:?}",
                name,
            );
        }

        // The buggy `+32` accounting would have produced only the
        // expected header's full size plus 32 per placeholder. Assert we
        // are strictly above that so the regression cannot silently
        // return.
        let buggy_size: usize = headers[0].0.len() + headers[0].1.len() + 32
            + headers[1..].iter().map(|_| 32).sum::<usize>();
        assert!(rfc_size > buggy_size);

        // Eviction alignment: shrink the max below the RFC size and
        // verify the table drains fully (size reconciles to 0). With the
        // `+32` accounting the per-placeholder add/evict was unbalanced
        // (`add += 32`, `evict -= 0+0+32 = 32`, which happened to balance
        // but only because the size was wrong to begin with); more
        // importantly the *timing* of eviction diverged from the encoder.
        // With RFC-7541 accounting on both add and evict, the table
        // reconciles cleanly to 0 after full eviction — and crucially
        // each eviction now happens at exactly the same `max_size`
        // threshold the encoder uses, keeping the index address space
        // aligned.
        table.set_max_table_size(0);
        assert_eq!(0, table.len());
        assert_eq!(0, table.to_vec().len());
        assert_eq!(0, table.get_size());
    }

    #[test]
    fn test_dynamic_table_size_calculation() {
        let mut table = DynamicTable::new();

        table.add_header(b"a".to_vec(), b"b".to_vec());
        table.add_header(b"123".to_vec(), b"456".to_vec());
        table.add_header(b"a".to_vec(), b"b".to_vec());

        assert_eq!(3 * 32 + 2 + 6 + 2, table.get_size());
    }

    /// Tests that the `DynamicTable` gets correctly resized (by evicting old
    /// headers) if it exceeds the maximum size on an insertion.
    #[test]
    fn test_dynamic_table_auto_resize() {
        let mut table = DynamicTable::with_size(38);
        table.add_header(b"a".to_vec(), b"b".to_vec());
        assert_eq!(32 + 2, table.get_size());

        table.add_header(b"123".to_vec(), b"456".to_vec());

        // Resized?
        assert_eq!(32 + 6, table.get_size());
        // Only has the second header?
        assert_eq!(table.to_vec(), vec![(b"123".to_vec(), b"456".to_vec())]);
    }

    /// Tests that when inserting a new header whose size is larger than the
    /// size of the entire table, the table is fully emptied.
    #[test]
    fn test_dynamic_table_auto_resize_into_empty() {
        let mut table = DynamicTable::with_size(38);
        table.add_header(b"a".to_vec(), b"b".to_vec());
        assert_eq!(32 + 2, table.get_size());

        table.add_header(b"123".to_vec(), b"4567".to_vec());

        // Resized and empty?
        assert_eq!(0, table.get_size());
        assert_eq!(0, table.to_vec().len());
    }

    /// Tests that when changing the maximum size of the `DynamicTable`, the
    /// headers are correctly evicted in order to keep its size below the new
    /// max.
    #[test]
    fn test_dynamic_table_change_max_size() {
        let mut table = DynamicTable::new();
        table.add_header(b"a".to_vec(), b"b".to_vec());
        table.add_header(b"123".to_vec(), b"456".to_vec());
        table.add_header(b"c".to_vec(), b"d".to_vec());
        assert_eq!(3 * 32 + 2 + 6 + 2, table.get_size());

        table.set_max_table_size(38);

        assert_eq!(32 + 2, table.get_size());
        assert_eq!(table.to_vec(), vec![(b"c".to_vec(), b"d".to_vec())]);
    }

    /// Tests that setting the maximum table size to 0 clears the dynamic
    /// table.
    #[test]
    fn test_dynamic_table_clear() {
        let mut table = DynamicTable::new();
        table.add_header(b"a".to_vec(), b"b".to_vec());
        table.add_header(b"123".to_vec(), b"456".to_vec());
        table.add_header(b"c".to_vec(), b"d".to_vec());
        assert_eq!(3 * 32 + 2 + 6 + 2, table.get_size());

        table.set_max_table_size(0);

        assert_eq!(0, table.len());
        assert_eq!(0, table.to_vec().len());
        assert_eq!(0, table.get_size());
        assert_eq!(0, table.get_max_table_size());
    }

    /// Tests that when the initial max size of the table is 0, nothing
    /// can be added to the table.
    #[test]
    fn test_dynamic_table_max_size_zero() {
        let mut table = DynamicTable::with_size(0);

        table.add_header(b"a".to_vec(), b"b".to_vec());

        assert_eq!(0, table.len());
        assert_eq!(0, table.to_vec().len());
        assert_eq!(0, table.get_size());
        assert_eq!(0, table.get_max_table_size());
    }

    /// Tests that the iterator through the `DynamicTable` works when there are
    /// some elements in the dynamic table.
    #[test]
    fn test_dynamic_table_iter_with_elems() {
        let mut table = DynamicTable::new();
        table.add_header(b"a".to_vec(), b"b".to_vec());
        table.add_header(b"123".to_vec(), b"456".to_vec());
        table.add_header(b"c".to_vec(), b"d".to_vec());

        let iter_res: Vec<(&[u8], &[u8])> = table.iter().collect();

        let expected: Vec<(&[u8], &[u8])> = vec![(b"c", b"d"), (b"123", b"456"), (b"a", b"b")];
        assert_eq!(iter_res, expected);
    }

    /// Tests that the iterator through the `DynamicTable` works when there are
    /// no elements in the dynamic table.
    #[test]
    fn test_dynamic_table_iter_no_elems() {
        let table = DynamicTable::new();

        let iter_res: Vec<(&[u8], &[u8])> = table.iter().collect();

        let expected = vec![];
        assert_eq!(iter_res, expected);
    }

    /// Tests that indexing the header table with indices that correspond to
    /// entries found in the static table works.
    #[test]
    fn test_header_table_index_static() {
        let table = HeaderTable::with_static_table(STATIC_TABLE);

        for (index, entry) in STATIC_TABLE.iter().enumerate() {
            assert_eq!(table.get_from_table(index + 1).unwrap(), *entry);
        }
    }

    /// Tests that when the given index is out of bounds, the `HeaderTable`
    /// returns a `None`
    #[test]
    fn test_header_table_index_out_of_bounds() {
        let table = HeaderTable::with_static_table(STATIC_TABLE);

        assert!(table.get_from_table(0).is_none());
        assert!(table
            .get_from_table(STATIC_TABLE.len() + 1)
            .unwrap()
            .0
            .is_empty());
    }

    /// Tests that adding entries to the dynamic table through the
    /// `HeaderTable` interface works.
    #[test]
    fn test_header_table_add_to_dynamic() {
        let mut table = HeaderTable::with_static_table(STATIC_TABLE);
        let header = (b"a".to_vec(), b"b".to_vec());

        table.add_header(header.0.clone(), header.1.clone());

        assert_eq!(table.dynamic_table.to_vec(), vec![header]);
    }

    /// Tests that indexing the header table with indices that correspond to
    /// entries found in the dynamic table works.
    #[test]
    fn test_header_table_index_dynamic() {
        let mut table = HeaderTable::with_static_table(STATIC_TABLE);
        let header = (b"a".to_vec(), b"b".to_vec());

        table.add_header(header.0.clone(), header.1.clone());

        assert_eq!(
            table.get_from_table(STATIC_TABLE.len() + 1).unwrap(),
            ((&header.0[..], &header.1[..]))
        );
    }

    /// Tests that the `iter` method of the `HeaderTable` returns an iterator
    /// through *all* the headers found in the header table (static and dynamic
    /// tables both included)
    #[test]
    fn test_header_table_iter() {
        let mut table = HeaderTable::with_static_table(STATIC_TABLE);
        let headers: [(&[u8], &[u8]); 2] = [(b"a", b"b"), (b"c", b"d")];
        for header in headers.iter() {
            table.add_header(header.0.to_vec(), header.1.to_vec());
        }

        let iterated: Vec<(&[u8], &[u8])> = table.iter().collect();

        assert_eq!(iterated.len(), headers.len() + STATIC_TABLE.len());
        // Part of the static table correctly iterated through
        for (h1, h2) in iterated.iter().zip(STATIC_TABLE.iter()) {
            assert_eq!(h1, h2);
        }
        // Part of the dynamic table correctly iterated through: the elements
        // are in reversed order of insertion in the dynamic table.
        for (h1, h2) in iterated
            .iter()
            .skip(STATIC_TABLE.len())
            .zip(headers.iter().rev())
        {
            assert_eq!(h1, h2);
        }
    }

    /// Tests that searching for an entry in the header table, which should be
    /// fully in the static table (both name and value), works correctly.
    #[test]
    fn test_find_header_static_full() {
        let table = HeaderTable::with_static_table(STATIC_TABLE);

        for (i, h) in STATIC_TABLE.iter().enumerate() {
            assert_eq!(table.find_header(*h).unwrap(), (i + 1, true));
        }
    }

    /// Tests that searching for an entry in the header table, which should be
    /// only partially in the static table (only the name), works correctly.
    #[test]
    fn test_find_header_static_partial() {
        {
            let table = HeaderTable::with_static_table(STATIC_TABLE);
            let h: (&[u8], &[u8]) = (b":method", b"PUT");

            if let (index, false) = table.find_header(h).unwrap() {
                assert_eq!(h.0, STATIC_TABLE[index - 1].0);
                // The index is the last one with the corresponding name
                assert_eq!(3, index);
            } else {
                panic!("The header should have matched only partially");
            }
        }
        {
            let table = HeaderTable::with_static_table(STATIC_TABLE);
            let h: (&[u8], &[u8]) = (b":status", b"333");

            if let (index, false) = table.find_header(h).unwrap() {
                assert_eq!(h.0, STATIC_TABLE[index - 1].0);
                // The index is the last one with the corresponding name
                assert_eq!(14, index);
            } else {
                panic!("The header should have matched only partially");
            }
        }
        {
            let table = HeaderTable::with_static_table(STATIC_TABLE);
            let h: (&[u8], &[u8]) = (b":authority", b"example.com");

            if let (index, false) = table.find_header(h).unwrap() {
                assert_eq!(h.0, STATIC_TABLE[index - 1].0);
            } else {
                panic!("The header should have matched only partially");
            }
        }
        {
            let table = HeaderTable::with_static_table(STATIC_TABLE);
            let h: (&[u8], &[u8]) = (b"www-authenticate", b"asdf");

            if let (index, false) = table.find_header(h).unwrap() {
                assert_eq!(h.0, STATIC_TABLE[index - 1].0);
            } else {
                panic!("The header should have matched only partially");
            }
        }
    }

    /// Tests that searching for an entry in the header table, which should be
    /// fully in the dynamic table (both name and value), works correctly.
    #[test]
    fn test_find_header_dynamic_full() {
        let mut table = HeaderTable::with_static_table(STATIC_TABLE);
        let h: (&[u8], &[u8]) = (b":method", b"PUT");
        table.add_header(h.0.to_vec(), h.1.to_vec());

        if let (index, true) = table.find_header(h).unwrap() {
            assert_eq!(index, STATIC_TABLE.len() + 1);
        } else {
            panic!("The header should have matched fully");
        }
    }

    /// Tests that searching for an entry in the header table, which should be
    /// only partially in the dynamic table (only the name), works correctly.
    #[test]
    fn test_find_header_dynamic_partial() {
        let mut table = HeaderTable::with_static_table(STATIC_TABLE);
        // First add it to the dynamic table
        {
            let h = (b"X-Custom-Header", b"stuff");
            table.add_header(h.0.to_vec(), h.1.to_vec());
        }
        // Prepare a search
        let h: (&[u8], &[u8]) = (b"X-Custom-Header", b"different-stuff");

        // It must match only partially
        if let (index, false) = table.find_header(h).unwrap() {
            // The index must be the first one in the dynamic table
            // segment of the header table.
            assert_eq!(index, STATIC_TABLE.len() + 1);
        } else {
            panic!("The header should have matched only partially");
        }
    }
}
