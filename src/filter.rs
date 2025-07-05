use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{
        Array, ArrowPrimitiveType, AsArray, BooleanArray, Float32Array, Float64Array, PrimitiveArray, RecordBatch, UInt64Array
    },
    buffer::NullBuffer,
    compute::{nullif, take_record_batch},
    datatypes::{DataType, Float32Type, Float64Type, Int32Type, Int64Type, Schema},
};

use num_traits::{Float, Num, Zero};

pub fn collect_deltas<T: Float, I: IntoIterator<Item = T>>(iter: I) -> Vec<T> {
    let mut last = None;
    let mut deltas = Vec::new();
    for v in iter {
        if let Some(last) = last {
            let delta = v - last;
            deltas.push(delta);
        }
        last = Some(v);
    }
    deltas.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    deltas
}

pub fn median<T: Float>(deltas: &[T]) -> Option<T> {
    let n = deltas.len();
    let median = if n <= 2 {
        deltas.first().copied()
    } else {
        let mid = n / 2;
        if n % 2 == 1 {
            Some(deltas[mid])
        } else {
            Some((deltas[mid] + deltas[mid + 1]) / (T::one() + T::one()))
        }
    };
    median
}

pub fn estimate_median_delta<T: Float, I: IntoIterator<Item = T>>(iter: I) -> (T, Vec<T>) {
    let deltas = collect_deltas(iter);
    let median_of = median(&deltas).unwrap_or_else(|| T::zero());
    let delta_below: Vec<T> = deltas.iter().copied().filter(|v| *v <= median_of).collect();
    let median_of = median(&delta_below).unwrap_or_else(|| T::zero());
    (median_of, deltas)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NullFillState {
    Unset,
    /// The coordinate is null at the start of the interval
    NullStart(usize),
    /// The coordinate is null at the end of the interval
    NullEnd(usize),
    /// Both coordinates are nulls
    NullBounded(usize, usize),
    Done,
}

struct NullTokenizer<'a> {
    array: &'a NullBuffer,
    i: usize,
    state: NullFillState,
    next_state: NullFillState,
}

impl<'a> Iterator for NullTokenizer<'a> {
    type Item = NullFillState;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.emit();
        if !matches!(state, NullFillState::Done) {
            Some(state)
        } else {
            None
        }
    }
}

impl<'a> NullTokenizer<'a> {
    fn new(array: &'a NullBuffer) -> Self {
        let mut this = Self {
            array,
            i: 0,
            state: NullFillState::Unset,
            next_state: NullFillState::Unset,
        };
        this.initialize_state();
        this
    }

    fn initialize_state(&mut self) {
        self.i = 0;
        let start_null = self.is_null();
        self.find_next_null();
        if !start_null {
            self.state = NullFillState::NullEnd(self.i);
            self.update_next_state();
        } else {
            let start = 0;
            if self.is_null() {
                self.state = NullFillState::NullBounded(start, self.i);
                self.update_next_state();
            }
            // TODO: Do we need to handle the `else` here?
        }
    }

    fn update_next_state(&mut self) {
        // self.state = self.next_state;
        let prev = self.i;
        self.find_next_null();
        let diff = self.i.saturating_sub(prev);
        if diff == 0 {
            // We are at the end
            self.next_state = NullFillState::Done
        } else if diff == 1 {
            // We stepped from one null to another
            let start = self.i;
            self.find_next_null();
            let end = self.i;
            if self.is_null() {
                self.next_state = NullFillState::NullBounded(start, end);
            } else {
                self.next_state = NullFillState::NullStart(start);
            }
        } else {
            // We stepped from one null into a run of values, this is probably not
            // right.
            eprintln!("Trip run")
        }
    }

    fn is_valid(&self) -> bool {
        self.array.is_valid(self.i)
    }

    fn is_null(&self) -> bool {
        self.array.is_null(self.i)
    }

    fn advance(&mut self) -> bool {
        if self.i < self.array.len().saturating_sub(1) {
            self.i += 1;
            return true;
        }
        return false;
    }

    fn find_next_null(&mut self) {
        self.advance();
        // eprintln!("Starting at {}", self.i);
        while self.i < self.array.len() && self.is_valid() {
            if !self.advance() {
                break;
            }
        }
        // eprintln!("Ending at {}", self.i);
    }

    fn emit(&mut self) -> NullFillState {
        let state = self.state;
        self.state = self.next_state;
        self.update_next_state();
        state
    }
}

pub fn fill_nulls_for<T: ArrowPrimitiveType>(
    data: &PrimitiveArray<T>,
    common_delta: T::Native,
) -> Vec<T::Native>
where
    T::Native: Float + Display,
{
    let Some(nulls) = data.nulls() else {
        let mut buffer: Vec<T::Native> = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            buffer.push(data.value(i));
        }
        return buffer;
    };

    let it = NullTokenizer::new(nulls);
    let n = data.len();
    let mut buffer: Vec<T::Native> = Vec::with_capacity(n);

    for null_span in it {
        match null_span {
            NullFillState::NullStart(start) => {
                let length = (n - start) - 1;
                let real_values = data.slice(start + 1, length);
                if length == 1 {
                    let val = real_values.value(0);
                    buffer.push(val - common_delta);
                    buffer.push(val);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    let val0 = real_values.value(0);
                    buffer.push(val0 - local_delta);
                    buffer.extend(real_values.iter().flatten());
                }
            },
            NullFillState::NullEnd(end) => {
                let start = 0;
                let length = end - start;
                let real_values = data.slice(start, length);
                if length == 1 {
                    let val = real_values.value(0);
                    buffer.push(val);
                    buffer.push(val + common_delta);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    buffer.extend(real_values.iter().flatten());
                    buffer.push(*buffer.last().unwrap() + local_delta);
                }
            },
            NullFillState::NullBounded(start, end) => {
                let length = (end - start) - 1;
                let real_values = data.slice(start + 1, length);
                if length == 1 {
                    let val = real_values.value(0);
                    buffer.push(val - common_delta);
                    buffer.push(val);
                    buffer.push(val + common_delta);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    let val0 = real_values.value(0);
                    buffer.push(val0 - local_delta);
                    buffer.extend(real_values.iter().flatten());
                    buffer.push(*buffer.last().unwrap() + local_delta);
                }
            },
            NullFillState::Unset | NullFillState::Done => {
                unimplemented!("These states should never occur")
            }
        }
    }
    buffer
}

fn _skip_zero_runs2<T: ArrowPrimitiveType>(array: &PrimitiveArray<T>) -> Vec<u64> where T::Native: Zero + PartialEq + Display {
    let z = T::Native::zero();
    let n = array.len();
    let n1 = n.saturating_sub(1);
    let mut was_zero = false;
    let mut acc = Vec::new();
    for (i, v) in array.iter().enumerate() {
        if let Some(v) = v {
            if v == z {
                if was_zero && ((i < n1 && array.value(i + 1) == z) || i == n1) {
                    // Skip, do not take values between two zeros
                } else {
                    acc.push(i as u64)
                }
                was_zero = true;
            } else {
                acc.push(i as u64);
                was_zero = false;
            }
        } else {
            acc.push(i as u64);
            was_zero = false;
        }
    }
    acc.into()
}

/// A type-generic filter to find indices where the value isn't in the middle of a run of zeros.
fn _skip_zero_runs<T: Num + Copy, I: Iterator<Item = Option<T>>>(iter: I) -> Vec<u64> {
    let mut points = Vec::new();
    let mut was_zero = false;
    let zero = T::zero();
    for (i, v) in iter.enumerate() {
        if v.unwrap_or(zero) == zero {
            if !was_zero {
                was_zero = true;
                points.push(i as u64);
            }
        } else {

            if was_zero && points.last().is_some_and(|j| *j != (i as u64) - 1) {
                points.push(i as u64 - 1);
            }
            was_zero = false;
            points.push(i as u64);
        }
    }
    points
}

pub fn find_where_not_zeros(array: &impl Array) -> Option<Vec<u64>> {
    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        Some(_skip_zero_runs2(array))
    } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        Some(_skip_zero_runs2(array))
    } else {
        None
    }
}

pub fn drop_where_column_is_zero(
    batch: &RecordBatch,
    column_index: usize,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let target_array = batch.column(column_index);
    if let Some(indices) = find_where_not_zeros(target_array) {
        take_record_batch(&batch, &UInt64Array::from(indices))
    } else {
        Ok(batch.clone())
    }
}

fn is_zero_pair_mask<T: ArrowPrimitiveType>(array: &PrimitiveArray<T>) -> BooleanArray where T::Native : Zero + PartialEq {
    let z = T::Native::zero();
    let n = array.len();
    let n1 = n.saturating_sub(1);
    let mut was_zero = false;
    let mut acc = Vec::new();
    for (i, v) in array.iter().enumerate() {
        if let Some(v) = v {
            if v == z {
                if was_zero || (i < n1 && array.value(i + 1) == z) {
                    acc.push(true);
                } else {
                    acc.push(false)
                }
                was_zero = true;
            } else {
                acc.push(false);
                was_zero = false;
            }
        } else {
            acc.push(false);
            was_zero = false;
        }
    }
    assert_eq!(acc.len(), n);
    acc.into()
}

pub fn nullify_at_zero(
    batch: &RecordBatch,
    column_index: usize,
    target_indices: &[usize],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let target_array = batch.column(column_index);
    let mask = match target_array.data_type() {
        DataType::Float32 => is_zero_pair_mask(target_array.as_primitive::<Float32Type>()),
        DataType::Float64 => is_zero_pair_mask(target_array.as_primitive::<Float64Type>()),
        DataType::Int32 => is_zero_pair_mask(target_array.as_primitive::<Int32Type>()),
        DataType::Int64 => is_zero_pair_mask(target_array.as_primitive::<Int64Type>()),
        _ => panic!("Unsupported data type {:?}", target_array.data_type()),
    };

    let (schema, mut cols, _row_count) = batch.clone().into_parts();

    let schema: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| Arc::new(f.as_ref().clone().with_nullable(true)))
        .collect();

    for (i, col) in cols.iter_mut().enumerate() {
        if !target_indices.contains(&i) {
            continue;
        }
        *col = nullif(col, &mask)?;
    }

    RecordBatch::try_new(Arc::new(Schema::new(schema)), cols)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zero_runs() {
        let data = Float64Array::from(vec![
            Some(0.0), // 0
            Some(101.0), // 1
            Some(101.01), // 2
            Some(101.02), // 3
            Some(0.0), // 4
            Some(0.0), // 5 This position should drop!
            Some(0.0), // 6
            Some(101.5), // 7
            Some(0.0), // 8
        ]);
        let indices = _skip_zero_runs2(&data);
        assert!(!indices.contains(&5), "index 5 should not be in {indices:?}");
        eprintln!("{indices:?}");
    }


    #[test]
    fn test_null_filling() {
        let data = Float64Array::from(vec![
            None,
            Some(101.0),
            Some(101.01),
            Some(101.02),
            None,
            None,
            Some(101.5),
            None,
        ]);
        let mut tokenizer = NullTokenizer::new(data.nulls().unwrap());
        let a = tokenizer.next().unwrap();
        assert_eq!(a, NullFillState::NullBounded(0, 4));
        let b = tokenizer.next().unwrap();
        assert_eq!(b, NullFillState::NullBounded(5, 7));

        let filled = fill_nulls_for(&data, 0.015);
        eprintln!("{filled:?}");
    }

    #[test]
    fn test_null_filling_tailed() {
        let data = Float64Array::from(vec![
            None,
            Some(101.0),
            Some(101.01),
            Some(101.02),
            None,
            None,
            Some(101.5),
        ]);
        let mut tokenizer = NullTokenizer::new(data.nulls().unwrap());

        let a = tokenizer.next().unwrap();
        assert_eq!(a, NullFillState::NullBounded(0, 4));
        let b = tokenizer.next().unwrap();
        assert_eq!(b, NullFillState::NullStart(5));

        let filled = fill_nulls_for(&data, 0.015);
        eprintln!("{filled:?}");
    }

    #[test]
    fn test_null_filling_no_prefix_suffix() {
        let data = Float64Array::from(vec![
            Some(101.0),
            Some(101.01),
            Some(101.02),
            None,
            None,
            Some(101.5),
        ]);
        let mut tokenizer = NullTokenizer::new(data.nulls().unwrap());

        let a = tokenizer.next().unwrap();

        assert_eq!(a, NullFillState::NullEnd(3));
        let b = tokenizer.next().unwrap();
        assert_eq!(b, NullFillState::NullStart(4));

        let filled = fill_nulls_for(&data, 0.015);
        eprintln!("{filled:?}");
    }
}
