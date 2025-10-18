use std::collections::HashSet;

use arrow::array::AsArray;
use identity_hash::BuildIdentityHasher;
use mzpeaks::coordinate::{Span1D, SimpleInterval};

use super::index::SpanDynNumeric;


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
    pub range: SimpleInterval<u64>,
    pub includes: Option<HashSet<u64, BuildIdentityHasher<u64>>>,
}

impl From<SimpleInterval<u64>> for MaskSet {
    fn from(value: SimpleInterval<u64>) -> Self {
        Self::new(value, None)
    }
}

impl MaskSet {
    pub fn new(range: SimpleInterval<u64>, includes: Option<HashSet<u64, BuildIdentityHasher<u64>>>) -> Self {
        Self { range, includes }
    }

    pub fn split(&mut self) -> Option<Self> {
        let halfway = (self.range.end - self.range.start) / 2;
        if halfway < 2 {
            return None
        }

        self.range.end = self.range.start + halfway;
        let mut other = self.clone();
        other.range.start = self.end() + 1;

        // TODO: Maybe split the `includes` set evenly?

        Some(other)
    }
}

impl Span1D for MaskSet {
    type DimType = u64;

    fn start(&self) -> Self::DimType {
        self.range.start
    }

    fn end(&self) -> Self::DimType {
        self.range.end
    }

    fn contains(&self, i: &Self::DimType) -> bool {
        if !self.range.contains(i) {
            false
        } else if let Some(includes) = self.includes.as_ref() {
            includes.contains(i)
        } else {
            true
        }
    }
}

impl SpanDynNumeric for MaskSet {
    fn contains_dy(&self, array: &arrow::array::ArrayRef) -> arrow::array::BooleanArray {
        let mask = self.range.contains_dy(array);
        if let Some(includes) = self.includes.as_ref() {
            if let Some(arr) = array.as_primitive_opt::<arrow::datatypes::UInt64Type>() {
                let is_in: arrow::array::BooleanArray = arr.iter().map(|v| Some(v.is_some_and(|v| includes.contains(&v)))).collect();
                arrow::compute::and(&mask, &is_in).unwrap()
            }
            else if let Some(arr) = array.as_primitive_opt::<arrow::datatypes::UInt32Type>() {
                let is_in = arr.iter().map(|v| Some(v.is_some_and(|v| includes.contains(&(v as u64))))).collect();
                arrow::compute::and(&mask, &is_in).unwrap()
            } else {
                panic!("Unsupported data type {:?}", array.data_type())
            }
        } else {
            mask
        }
    }
}