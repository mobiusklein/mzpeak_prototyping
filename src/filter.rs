use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{
        Array, ArrowPrimitiveType, Float32Array, Float64Array, Int32Array, Int64Array, PrimitiveArray, RecordBatch, UInt64Array
    }, buffer::NullBuffer, compute::{kernels::cmp::eq, nullif, take_record_batch}, datatypes::{DataType, Schema}
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
    let median = if n <= 1 {
        deltas.first().copied()
    } else {
        let mid = n / 2;
        if n % 2 == 1 {
            Some(deltas[mid])
        } else {
            Some(((deltas[mid] + deltas[mid + 1])) / (T::one() + T::one()))
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
    /// The coordinate is null, at the start of the interval
    NullStart(usize),
    /// The coordinate is null, at the end of the interval
    NullEnd(usize),
    /// Both coordinates are nulls
    NullBounded(usize, usize),
    Done,
}


struct NullTokenizer<'a, T: ArrowPrimitiveType> {
    array: &'a PrimitiveArray<T>,
    i: usize,
    state: NullFillState,
    next_state: NullFillState,
}

impl<'a, T: ArrowPrimitiveType> NullTokenizer<'a, T> {
    fn new(array: &'a PrimitiveArray<T>) -> Self {
        let mut this = Self { array, i: 0, state: NullFillState::Unset, next_state: NullFillState::Unset };
        this.initialize_state();
        this
    }

    fn initialize_state(&mut self) {
        self.i = 0;
        self.find_next_null();
        if self.i != 0 {
            self.state = NullFillState::NullEnd(self.i);
            if self.is_next_null() {
                if self.advance() {
                    let start = self.i;
                    self.find_next_null();
                    if self.is_null() {
                        self.next_state = NullFillState::NullBounded(start, self.i)
                    } else {
                        self.next_state = NullFillState::NullStart(start)
                    }
                }
            }
        }
    }

    fn update_next_state(&mut self) {
        // self.state = self.next_state;
        let prev = self.i;
        self.find_next_null();
        let diff = self.i - prev;
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
        if self.i < self.array.len() {
            self.i += 1;
            return true
        }
        return false;
    }

    fn find_next_null(&mut self) {
        eprintln!("Starting at {}", self.i);
        while self.i < self.array.len() && self.is_valid() {
            self.i += 1;
        }
        eprintln!("Ending at {}", self.i);
    }

    fn is_next_valid(&self) -> bool {
        self.array.is_valid(self.i + 1)
    }

    fn is_next_null(&self) -> bool {
        self.array.is_null(self.i + 1)
    }

    fn emit(&mut self) -> NullFillState {
        let state = self.state;
        self.update_next_state();
        state
    }
}


pub fn fill_nulls_for<T: ArrowPrimitiveType>(data: &PrimitiveArray<T>, common_delta: T::Native) -> Vec<T::Native> where T::Native : Float + Display {
    let Some(nulls) = data.nulls() else {
        let mut buffer: Vec<T::Native> = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            buffer.push(data.value(i));
        }
        return buffer
    };

    let it = nulls.iter().enumerate().filter(|(_i, f)| !*f);
    let mut last_null = Some(0);
    let mut buffer: Vec<T::Native> = Vec::with_capacity(data.len());
    let mut backfill = None;
    for (i, _is_not_null) in it {
        if let Some(last_null) = last_null {
            let diff = i - last_null;
            eprintln!("{i} | {last_null} | {backfill:?}");
            if diff == 0 {
                continue;
            }
            if diff == 1 {
                // run of null, start new frame
                eprintln!("Null run at {i} ({} @ {})", buffer.last().unwrap(), buffer.len());
                backfill = Some(buffer.len());
                buffer.push(T::Native::zero());
            } else if diff == 2 {
                eprintln!("Point from {last_null} to {i} ({})", buffer.last().unwrap());
                buffer.push(data.value(i - 1));
                buffer.push(*buffer.last().unwrap() + common_delta);
                if let Some(backfill_idx) = backfill {
                    // eprintln!("Backfilling {backfill_idx}");
                    buffer[backfill_idx] = buffer[backfill_idx + 1] - common_delta;
                    backfill = None;
                }
            } else {
                eprintln!("Filling span from {} to {}", last_null + 1, i);
                // a span of non-null values
                for j in (last_null + 1)..i {
                    buffer.push(data.value(j));
                }
                let local_delta = estimate_median_delta(buffer[last_null + 1..].iter().copied()).0;
                eprintln!("Local delta = {local_delta}");
                let next_val = data.value(i.saturating_sub(1)) + local_delta;
                buffer.push(next_val);
                if let Some(backfill_idx) = backfill {
                    // eprintln!("Backfilling {backfill_idx}");
                    buffer[backfill_idx] = buffer[backfill_idx + 1] - local_delta;
                    backfill = None;
                }
            }
        }
        last_null = Some(i);
    }

    if let Some(backfill_idx) = backfill {
        let i = data.len();
        let diff = i - backfill_idx;
        eprintln!("Backfilling {backfill_idx}");
        eprintln!("{i} | {last_null:?} | {backfill:?}");

        backfill = None;
        if diff == 2 {
            eprintln!("Point from {backfill_idx} to {i} ({})", buffer.last().unwrap());
            buffer.push(data.value(i - 1));
            buffer.push(*buffer.last().unwrap() + common_delta);
            if let Some(backfill_idx) = backfill {
                eprintln!("Backfilling {backfill_idx}");
                buffer[backfill_idx] = buffer[backfill_idx + 1] - common_delta;
            }
        } else if diff > 0 {
            eprintln!("Filling span from {} to {}", backfill_idx + 1, i);
            // a span of non-null values
            for j in (backfill_idx + 1)..i {
                buffer.push(data.value(j));
            }
            let local_delta = estimate_median_delta(buffer[backfill_idx + 1..].iter().copied()).0;
            // eprintln!("Local delta = {local_delta}");
            let next_val = data.value(i.saturating_sub(1)) + local_delta;
            buffer.push(next_val);
            if let Some(backfill_idx) = backfill {
                eprintln!("Backfilling {backfill_idx}");
                buffer[backfill_idx] = buffer[backfill_idx + 1] - local_delta;
            }
        }
    }
    buffer
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
            if was_zero {
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
        Some(_skip_zero_runs(array.iter()))
    } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        Some(_skip_zero_runs(array.iter()))
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

pub fn nullify_at_zero(
    batch: &RecordBatch,
    column_index: usize,
    skip_indices: &[usize],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let target_array = batch.column(column_index);
    let mask = match target_array.data_type() {
        DataType::Float32 => eq(target_array, &Float32Array::new_scalar(0.0))?,
        DataType::Float64 => eq(target_array, &Float64Array::new_scalar(0.0))?,
        DataType::Int32 => eq(target_array, &Int32Array::new_scalar(0))?,
        DataType::Int64 => eq(target_array, &Int64Array::new_scalar(0))?,
        _ => panic!("Unsupported data type {:?}", target_array.data_type()),
    };

    let (schema, mut cols, _row_count) = batch.clone().into_parts();

    let schema: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| Arc::new(f.as_ref().clone().with_nullable(true)))
        .collect();

    for (i, col) in cols.iter_mut().enumerate() {
        if skip_indices.contains(&i) {
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
        let data = data;
        let filled = fill_nulls_for(&data, 0.02f64);
        eprintln!("{filled:?}")
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
        let data = data;
        // let filled = fill_nulls_for(&data, 0.02f64);
        // eprintln!("{filled:?}")
        let mut tokenizer = NullTokenizer::new(&data);
        loop {
            let out = tokenizer.emit();
            match out {
                NullFillState::Unset => eprintln!("{out:?}"),
                NullFillState::NullStart(_) => eprintln!("{out:?}"),
                NullFillState::NullEnd(_) => eprintln!("{out:?}"),
                NullFillState::NullBounded(_, _) => eprintln!("{out:?}"),
                NullFillState::Done => break,
            }
        }
    }
}