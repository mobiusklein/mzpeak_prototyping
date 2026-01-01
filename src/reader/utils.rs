use std::collections::HashSet;

use arrow::{array::{AsArray, RecordBatch}, error::ArrowError};
use identity_hash::BuildIdentityHasher;
use mzpeaks::coordinate::{SimpleInterval, Span1D};

use super::index::SpanDynNumeric;

pub type BatchIterator<'a> = Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + 'a>;

#[derive(Default, Debug)]
pub(crate) struct OneCache<T: PartialEq + Eq, U> {
    last_key: Option<T>,
    last_value: Option<U>,
}

impl<T: PartialEq + Eq, U> OneCache<T, U> {
    pub(crate) fn get<F: FnOnce() -> U>(&mut self, key: T, callback: F) -> &U {
        let key = Some(key);
        if self.last_key == key {
            return self.last_value.as_ref().unwrap();
        } else {
            self.last_key = key;
            self.last_value = Some(callback());
            return self.last_value.as_ref().unwrap();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaskSet {
    pub index_range: SimpleInterval<u64>,
    pub sparse_includes: Option<HashSet<u64, BuildIdentityHasher<u64>>>,
}

impl From<SimpleInterval<u64>> for MaskSet {
    fn from(value: SimpleInterval<u64>) -> Self {
        Self::new(value, None)
    }
}

impl MaskSet {
    pub fn new(
        range: SimpleInterval<u64>,
        includes: Option<HashSet<u64, BuildIdentityHasher<u64>>>,
    ) -> Self {
        Self { index_range: range, sparse_includes: includes }
    }

    pub fn split(&mut self) -> Option<Self> {
        let halfway = (self.index_range.end - self.index_range.start) / 2;
        if halfway < 2 {
            return None;
        }

        self.index_range.end = self.index_range.start + halfway;
        let mut other = Self::new(self.index_range.clone(), None);
        other.index_range.start = self.end() + 1;
        if let Some(includes) = self.sparse_includes.as_mut() {
            let mut other_includes = HashSet::with_capacity_and_hasher(includes.len() / 2, Default::default());
            includes.retain(|v| {
                if other.index_range.contains(v) {
                    other_includes.insert(*v);
                    false
                } else {
                    true
                }
            });

            other.sparse_includes = Some(other_includes);
        }
        Some(other)
    }

    pub fn empty() -> Self {
        Self::new(SimpleInterval::new(u64::MAX, u64::MAX), None)
    }

    pub fn intersect(&self, other: &Self) -> Self {
        if !self.overlaps(other) {
            Self::empty()
        } else {
            let start = self.start().max(other.start());
            let end = self.end().min(other.end());
            let range = SimpleInterval::new(start, end);
            let includes = match (self.sparse_includes.as_ref(), other.sparse_includes.as_ref()) {
                (None, None) => None,
                (None, Some(b)) => Some(b.iter().filter(|i| range.contains(*i)).copied().collect()),
                (Some(a), None) => Some(a.iter().filter(|i| range.contains(*i)).copied().collect()),
                (Some(a), Some(b)) => Some(
                    a.intersection(b)
                        .into_iter()
                        .filter(|i| range.contains(i))
                        .copied()
                        .collect(),
                ),
            };
            Self::new(range, includes)
        }
    }
}

impl Span1D for MaskSet {
    type DimType = u64;

    fn start(&self) -> Self::DimType {
        self.index_range.start
    }

    fn end(&self) -> Self::DimType {
        self.index_range.end
    }

    fn contains(&self, i: &Self::DimType) -> bool {
        if !self.index_range.contains(i) {
            false
        } else if let Some(includes) = self.sparse_includes.as_ref() {
            includes.contains(i)
        } else {
            true
        }
    }
}

impl SpanDynNumeric for MaskSet {
    fn contains_dy(&self, array: &arrow::array::ArrayRef) -> arrow::array::BooleanArray {
        let mask = self.index_range.contains_dy(array);
        if let Some(includes) = self.sparse_includes.as_ref() {
            if let Some(arr) = array.as_primitive_opt::<arrow::datatypes::UInt64Type>() {
                let is_in: arrow::array::BooleanArray = arr
                    .iter()
                    .map(|v| Some(v.is_some_and(|v| includes.contains(&v))))
                    .collect();
                arrow::compute::and(&mask, &is_in).unwrap()
            } else if let Some(arr) = array.as_primitive_opt::<arrow::datatypes::UInt32Type>() {
                let is_in = arr
                    .iter()
                    .map(|v| Some(v.is_some_and(|v| includes.contains(&(v as u64)))))
                    .collect();
                arrow::compute::and(&mask, &is_in).unwrap()
            } else {
                panic!("Unsupported data type {:?}", array.data_type())
            }
        } else {
            mask
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mask_set_new() {
        let mask: MaskSet = SimpleInterval::new(10, 30).into();
        assert!(mask.contains(&25));
        assert!(!mask.contains(&5));
        assert!(mask.sparse_includes.is_none());

        let mask = MaskSet::new((10u64..30).into(), Some(HashSet::from_iter([15, 27])));
        assert!(!mask.contains(&25));
        assert!(!mask.contains(&5));
        assert!(mask.contains(&15));
        assert!(mask.contains(&27));
    }
}
