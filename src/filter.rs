use std::{fmt::Debug, ops::AddAssign, sync::Arc};

use arrow::{
    array::{
        Array, ArrowPrimitiveType, AsArray, BooleanArray, Float32Array, Float64Array,
        PrimitiveArray, RecordBatch, UInt64Array,
    },
    buffer::NullBuffer,
    compute::{nullif, take_record_batch},
    datatypes::{DataType, Float32Type, Float64Type, Int32Type, Int64Type, Schema},
};

use num_traits::{Float, NumCast, One, Zero};

pub fn collect_deltas<T: Float, I: IntoIterator<Item = T>>(iter: I, sort: bool) -> Vec<T> {
    let mut last = None;
    let mut deltas = Vec::new();
    for v in iter {
        if let Some(last) = last {
            let delta = v - last;
            deltas.push(delta);
        }
        last = Some(v);
    }
    if sort {
        deltas.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    }
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
    let deltas = collect_deltas(iter, true);
    let median_of = median(&deltas).unwrap_or_else(|| T::zero());
    let delta_below: Vec<T> = deltas.iter().copied().filter(|v| *v <= median_of).collect();
    let median_of = median(&delta_below).unwrap_or_else(|| T::zero());
    (median_of, deltas)
}

pub fn fit_delta_model<
    T: Float
        + num_traits::One
        + num_traits::Zero
        + nalgebra::Scalar
        + nalgebra::ClosedAddAssign
        + nalgebra::ClosedMulAssign
        + nalgebra::ComplexField,
>(
    x_data: &[T],
    dx_data: &[T],
    chol_weights: Option<&[T]>,
    rank: usize,
) -> Result<Vec<T>, &'static str>
where
    T::RealField: Float + One + Zero,
{
    let xmat = nalgebra::DMatrix::from_fn(x_data.len(), rank + 1, |i, j| {
        if j == 0 {
            T::one()
        } else {
            nalgebra::ComplexField::powi(x_data[i], j as i32)
        }
    });

    let y = nalgebra::DMatrix::from_column_slice(dx_data.len(), 1, &dx_data);

    if let Some(weights) = chol_weights {
        let weights = nalgebra::DVectorView::from_slice_generic(weights, nalgebra::Dyn(weights.len()), nalgebra::Const::<1>);
        let chol_w_x = xmat.map_with_location(|i, _, x| {
            weights[i] * x
        });
        let chol_w_y = y.map_with_location(|i, _, v| v * weights[i]);

        let qr = chol_w_x.qr();
        let v = qr.q().transpose() * chol_w_y;
        let r = qr.r();

        let sol = r.solve_upper_triangular(&v).unwrap();
        Ok(sol.data.into())
    } else {
        let qr = xmat.qr();

        let v = qr.q().transpose() * y;
        let r = qr.r();
        let sol = r.solve_upper_triangular(&v).unwrap();
        Ok(sol.data.into())
    }
}

pub fn select_delta_model<
    T: Float
        + nalgebra::Scalar
        + nalgebra::ClosedAddAssign
        + nalgebra::ClosedMulAssign
        + nalgebra::ComplexField,
>(
    mz_array: &[T],
    weights: Option<&[T]>,
) -> Vec<f64>
where
    T::RealField: Float,
{
    let deltas = collect_deltas(mz_array.iter().copied(), false);
    let mut deltas_sorted = deltas.clone();
    deltas_sorted.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    let median_of = median(&deltas_sorted).unwrap_or_else(|| T::zero());
    let delta_below: Vec<T> = deltas_sorted.iter().copied().filter(|v| *v <= median_of).collect();
    let median_of = median(&delta_below).unwrap_or_else(|| T::zero());
    let constant_model = ConstantDeltaModel::from(median_of);

    let e_const = constant_model.mean_squared_error(mz_array, &deltas);

    if mz_array.len() > 10_000 {
        log::debug!("Fitting model with {} data points", mz_array.len());
    }
    let reg_model = RegressionDeltaModel::fit(
        &mz_array[1..],
        &deltas,
        T::from(1.0).unwrap(),
        weights.map(|w| &w[1..]),
    );
    let e_reg = reg_model.mean_squared_error(mz_array, &deltas);
    if e_const < (e_reg / T::from(10.0).unwrap()) {
        log::trace!("Using constant model");
        log::trace!("Constant delta error = {e_const}");
        log::trace!("Regress  delta error = {e_reg}");

        constant_model.to_float64_array().values().to_vec()
    } else {
        reg_model.to_float64_array().values().to_vec()
    }
}

pub trait MZDeltaModel<T: Float + AddAssign> {
    fn predict<U: Float + AddAssign + NumCast>(&self, mz: U) -> U;

    fn fit(mz_array: &[T], deltas: &[T], threshold: T, weights: Option<&[T]>) -> Self;

    fn mean_squared_error(&self, mzs: &[T], deltas: &[T]) -> T {
        let mut acc = T::zero();
        for (mz, delta) in mzs.iter().zip(deltas) {
            let err = (self.predict(*mz) - *delta).powi(2);
            acc += err;
        }
        acc
    }

    fn to_float64_array(&self) -> Float64Array;

    fn from_float64_array(data: &Float64Array) -> Self;
}

#[derive(Debug, Default, Clone)]
pub struct ConstantDeltaModel<T: Float + AddAssign> {
    pub delta: T,
}

impl<T: Float + AddAssign> From<T> for ConstantDeltaModel<T> {
    fn from(value: T) -> Self {
        Self { delta: value }
    }
}

impl<T: Float + AddAssign + NumCast> MZDeltaModel<T> for ConstantDeltaModel<T> {
    #[inline]
    fn predict<U: Float + AddAssign + NumCast>(&self, _mz: U) -> U {
        U::from(self.delta).unwrap()
    }

    fn fit(_mz_array: &[T], deltas: &[T], _threshold: T, _weights: Option<&[T]>) -> Self {
        let (delta, _) = estimate_median_delta(deltas.iter().copied());
        Self { delta }
    }

    fn to_float64_array(&self) -> Float64Array {
        Float64Array::from_value(NumCast::from(self.delta).unwrap(), 1)
    }

    fn from_float64_array(data: &Float64Array) -> Self {
        Self::from(T::from(data.value(0)).unwrap())
    }
}

#[derive(Debug, Default, Clone)]
pub struct RegressionDeltaModel<T: Float + AddAssign> {
    pub beta: Vec<T>,
}

impl<T: Float + AddAssign> From<Vec<T>> for RegressionDeltaModel<T> {
    fn from(value: Vec<T>) -> Self {
        Self { beta: value }
    }
}

impl<T: Float + AddAssign + nalgebra::ComplexField> MZDeltaModel<T> for RegressionDeltaModel<T>
where
    T::RealField: Float,
{
    #[inline]
    fn predict<U: Float + AddAssign + NumCast>(&self, mz: U) -> U {
        let mz = T::from(mz).unwrap();
        let mut acc = T::zero();
        for (i, b) in self.beta.iter().copied().enumerate() {
            if i == 0 {
                acc += b;
            } else {
                acc += b * Float::powi(mz, i as i32);
            }
        }
        U::from(acc).unwrap()
    }

    fn fit(mz_array: &[T], deltas: &[T], threshold: T, weights: Option<&[T]>) -> Self {
        if let Some(weights) = weights {
            let mut sel_deltas = Vec::with_capacity(deltas.len());
            let mut sel_mzs = Vec::with_capacity(mz_array.len());
            let mut sel_weights = Vec::with_capacity(weights.len());
            for ((delta, mz), weight) in deltas.iter().zip(mz_array).zip(weights) {
                if *delta > threshold {
                    continue;
                }
                sel_deltas.push(*delta);
                sel_mzs.push(*mz);
                sel_weights.push(*weight);
            }
            let x = fit_delta_model(&sel_mzs, &sel_deltas, Some(&sel_weights), 2).unwrap();
            Self::from(x)
        } else {
            let (sel_deltas, sel_mzs): (Vec<_>, Vec<_>) = deltas
                .iter()
                .copied()
                .zip(mz_array.iter().copied())
                .filter(|(d, _m)| *d <= threshold)
                .collect();
            let x = fit_delta_model(&sel_mzs, &sel_deltas, None, 2).unwrap();
            Self::from(x)
        }
    }

    fn to_float64_array(&self) -> Float64Array {
        Float64Array::from_iter_values(self.beta.iter().map(|v| NumCast::from(*v).unwrap()))
    }

    fn from_float64_array(data: &Float64Array) -> Self {
        let vals: Vec<T> = data.iter().flatten().flat_map(|v| T::from(v)).collect();
        Self::from(vals)
    }
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
            // right. We don't have a way to back-fill these accurately.
            eprintln!("Trip run: {prev} -> {}", self.i);
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

pub fn fill_nulls_for<T: ArrowPrimitiveType, F: MZDeltaModel<T::Native> + Debug>(
    data: &PrimitiveArray<T>,
    common_delta: &F,
) -> Vec<T::Native>
where
    T::Native: Float + AddAssign,
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
                    let delta_at = common_delta.predict(val);
                    buffer.push(val - delta_at);
                    buffer.push(val);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    let val0 = real_values.value(0);
                    buffer.push(val0 - local_delta);
                    buffer.extend(real_values.iter().flatten());
                }
            }
            NullFillState::NullEnd(end) => {
                let start = 0;
                let length = end - start;
                let real_values = data.slice(start, length);
                if length == 1 {
                    let val = real_values.value(0);
                    buffer.push(val);
                    let delta_at = common_delta.predict(val);
                    buffer.push(val + delta_at);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    buffer.extend(real_values.iter().flatten());
                    buffer.push(*buffer.last().unwrap() + local_delta);
                }
            }
            NullFillState::NullBounded(start, end) => {
                let length = (end - start) - 1;
                let real_values = data.slice(start + 1, length);
                if length == 1 {
                    let val = real_values.value(0);
                    let delta_at = common_delta.predict(val);
                    buffer.push(val - delta_at);
                    buffer.push(val);
                    buffer.push(val + delta_at);
                } else {
                    let (local_delta, _) = estimate_median_delta(real_values.iter().flatten());
                    let val0 = real_values.value(0);
                    buffer.push(val0 - local_delta);
                    buffer.extend(real_values.iter().flatten());
                    buffer.push(*buffer.last().unwrap() + local_delta);
                }
            }
            NullFillState::Unset | NullFillState::Done => {
                unimplemented!("These states should never occur")
            }
        }
    }
    buffer
}

pub(crate) fn _skip_zero_runs_iter<T: ArrowPrimitiveType, I: Iterator<Item=Option<T::Native>>>(iter: I, n: usize) -> Vec<u64> where T::Native: Zero + PartialEq {
    let z = T::Native::zero();
    let n1 = n.saturating_sub(1);
    let mut was_zero = false;
    let mut acc = Vec::new();
    let mut iter = iter.peekable();
    let mut i = 0;
    while let Some(v) = iter.next() {
        if let Some(v) = v {
            if v == z {
                if (was_zero || acc.is_empty()) && ((i < n1 && iter.peek().unwrap().unwrap() == z) || i == n1)
                {
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
        i += 1;
    }
    acc.into()
}

/// A type-generic filter to find indices where the value isn't in the middle of a run of zeros.
pub(crate) fn _skip_zero_runs_gen<T: ArrowPrimitiveType>(array: &PrimitiveArray<T>) -> Vec<u64>
where
    T::Native: Zero + PartialEq,
{
    let z = T::Native::zero();
    let n = array.len();
    let n1 = n.saturating_sub(1);
    let mut was_zero = false;
    let mut acc = Vec::new();
    for (i, v) in array.iter().enumerate() {
        if let Some(v) = v {
            if v == z {
                if (was_zero || acc.is_empty()) && ((i < n1 && array.value(i + 1) == z) || i == n1)
                {
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

pub fn find_where_not_zeros(array: &impl Array) -> Option<Vec<u64>> {
    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        Some(_skip_zero_runs_gen(array))
    } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        Some(_skip_zero_runs_gen(array))
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

fn is_zero_pair_mask<T: ArrowPrimitiveType>(array: &PrimitiveArray<T>) -> BooleanArray
where
    T::Native: Zero + PartialEq,
{
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
            Some(0.0),    // 0
            Some(101.0),  // 1
            Some(101.01), // 2
            Some(101.02), // 3
            Some(0.0),    // 4
            Some(0.0),    // 5 This position should drop!
            Some(0.0),    // 6
            Some(101.5),  // 7
            Some(0.0),    // 8
        ]);
        let indices = _skip_zero_runs_gen(&data);
        assert!(
            !indices.contains(&5),
            "index 5 should not be in {indices:?}"
        );
        eprintln!("{indices:?}");
    }

    #[test]
    fn test_zero_runs_tailed() {
        let data = Float64Array::from(vec![
            0., 0., 0., 0., 0., 0., 0., 0., 14627.247, 24097.691, 27908.545, 23742.236, 14179.794,
            0., 0., 0., 0., 0., 0., 0., 0., 13766.747, 24281.814, 29634.895, 26867.996, 17907.01,
            0., 0., 0., 0., 0., 0., 0., 0.,
        ]);
        let indices = _skip_zero_runs_gen(&data);
        assert!(!indices.contains(&5),);
        assert!(!indices.contains(&6),);
        assert!(indices.contains(&7));
        assert!(indices.contains(&26));
        assert!(!indices.contains(&27));
        eprintln!("{indices:?}")
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

        let filled = fill_nulls_for(&data, &ConstantDeltaModel::from(0.015f64));
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

        let filled = fill_nulls_for(&data, &ConstantDeltaModel::from(0.015f64));
        eprintln!("{filled:?}");
    }

    #[test]
    fn test_null_filling_long_prefix() {
        let data = Float64Array::from(vec![
            None,
            None,
            Some(101.0),
            Some(101.01),
            Some(101.02),
            None,
            None,
            Some(101.5),
        ]);
        let tokenizer = NullTokenizer::new(data.nulls().unwrap());
        for tok in tokenizer {
            eprintln!("{tok:?}");
        }
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

        let filled = fill_nulls_for(&data, &ConstantDeltaModel::from(0.015f64));
        eprintln!("{filled:?}");
    }

    #[test]
    fn test_quadratic_reg_delta() {
        let data = [101.06132409, 101.09376874, 101.0951795 ,
        101.09659027, 107.05492213, 107.0563739 , 107.05782566,
        109.09261967, 109.09408518, 109.0955507 , 111.10959153,
        111.11107053, 111.11254953, 111.11402853, 111.11550756,
        111.11698659, 114.09624053, 114.09773927, 114.09923802,
        118.57866172, 118.58018962, 118.58171752, 141.16369557,
        141.16536264, 141.16702971, 147.05867321, 147.06037473,
        147.06207625, 147.06377777, 147.06547932, 147.06718087,
        147.11312642, 147.11482825, 147.11653009, 155.18019904,
        155.18194692, 155.18369479, 179.1507967 , 179.15267472,
        179.15455274, 184.17350442, 184.17540859, 184.17731276,
        185.19172493, 185.19363435, 185.19554377, 189.10547029,
        189.10739978, 189.10932927, 189.12283599, 189.12476558,
        189.12669516, 193.05624078, 193.05819033, 193.06013987,
        199.07382632, 199.07580601, 199.07778571, 201.12607786,
        201.12806773, 201.1300576 , 202.16413005, 202.16612505,
        202.16812005, 203.1768535 , 203.1788535 , 203.18085349,
        205.19582167, 205.19783157, 205.19984147, 207.03091589,
        207.03293476, 207.03495363, 207.03697252, 207.03899141,
        217.99338891, 217.99546055, 217.99753218, 221.0493025 ,
        221.0513886 , 221.05347471, 221.07850869, 221.08059493,
        221.08268117, 221.08476742, 221.08685368, 221.08893995,
        221.09102622, 221.09311252, 221.09519882, 222.0810309 ,
        222.08312186, 222.08521283, 223.05648253, 223.05857809,
        223.06067364, 223.0627692 , 223.06486477, 223.06696035,
        233.08491357, 233.08705571, 233.08919786, 236.23369167,
        236.23584823, 236.23800479, 236.71268764, 236.71484639,
        236.71700513, 236.99556601, 236.99772605, 236.99988608,
        259.1692639 , 259.17152273, 259.17378155, 264.26281737,
        264.26509828, 264.26737919, 264.27650294, 264.27878391,
        264.28106488, 266.99551976, 266.99781243, 267.00010511,
        267.00239779, 267.00698319, 267.00927591, 267.01156864,
        268.00754074, 268.00983776, 268.01213478, 269.22633465,
        269.22863688, 269.23093911, 269.23784587, 269.24014815,
        269.24245043, 273.17984087, 273.18215995, 273.18447902,
        277.22290074, 277.22523691, 277.22757309, 279.22860439,
        279.23094901, 279.23329362, 281.03921324, 281.04156544,
        281.04391764, 281.04626985, 281.04862207, 281.0509743 ,
        281.05332654, 281.05567879, 281.05803105, 281.06038331,
        281.06273558, 281.06508788, 281.06744018, 283.1460001 ,
        283.1483611 , 283.1507221 , 285.05452014, 285.05688908,
        285.05925803, 285.22036925, 285.22273888, 285.22510852,
        287.23096619, 287.23334415, 287.23572212, 287.25474623,
        287.25712429, 287.25950236, 295.21618619, 295.21859698,
        295.22100778, 313.26092021, 313.26340359, 313.26588698,
        313.26837036, 313.27085377, 313.27333718, 313.28078747,
        313.28327093, 313.28575439, 313.28823786, 313.30562243,
        313.30810599, 313.31058955, 320.3006465 , 320.30315764,
        320.30566877, 323.26557174, 323.26809447, 323.2706172 ,
        327.19521407, 327.19775208, 327.20029009, 336.28514744,
        336.28772047, 336.29029349, 339.29455539, 339.29713991,
        339.29972442, 339.98755472, 339.99014187, 339.99272903,
        341.00505988, 341.0076509 , 341.01024192, 341.01283295,
        341.01542399, 341.01801504, 341.0206061 , 341.02319717,
        341.02578825, 341.02837934, 341.03097044, 341.03356155,
        341.03615267, 341.15017164, 341.15276321, 341.15535479,
        342.00592146, 342.00851628, 342.0111111 , 342.01370593,
        342.01630077, 342.01889561, 342.02149048, 342.02408535,
        343.00565108, 343.00824969, 343.0108483 , 343.01344692,
        343.01604555, 343.02384148, 343.02644016, 343.02903884,
        343.03163753, 343.30455461, 343.30715435, 343.30975409,
        343.42415247, 343.42675267, 343.42935286, 344.01985178,
        344.02245423, 344.02505668, 345.0042751 , 345.00688127,
        345.00948744, 355.06068829, 355.06333217, 355.06597605,
        355.06861994, 355.07126383, 355.07919559, 355.08183953,
        355.08448348, 355.08712743, 355.0897714 , 355.09241539,
        355.09505938, 355.28545238, 355.28809709, 355.29074181,
        356.0660677 , 356.06871532, 356.07136294, 356.07930585,
        356.08195352, 356.08460119, 357.05961189, 357.0622632 ,
        357.06491451, 357.07021716, 357.07286851, 357.07551986,
        357.33539999, 357.33805233, 357.34070466, 358.05454031,
        358.05719532, 358.05985032, 359.02160809, 359.02426668,
        359.02692526, 359.05085297, 359.05351166, 359.05617036,
        359.27155726, 359.27421677, 359.27687628, 361.26628114,
        361.26894802, 361.27161491, 362.05076607, 362.05343585,
        362.05610563, 369.11976431, 369.12246002, 369.12515574,
        370.35812774, 370.36082798, 370.36352821, 371.10646594,
        371.1091689 , 371.11187186, 371.1172778 , 371.1199808 ,
        371.12268379, 372.09910974, 372.10181631, 372.10452288,
        372.11805588, 372.12076253, 372.12346917, 375.09043076,
        375.09314819, 375.09586562, 385.27558554, 385.27833961,
        385.28109369, 385.28384777, 386.38075586, 386.38351388,
        386.38627191, 391.18349259, 391.1862677 , 391.18904281,
        392.37769454, 392.38047388, 392.38325323, 393.30932249,
        393.31210513, 393.31488777, 400.99314976, 400.99595946,
        400.99876915, 401.20952411, 401.21233456, 401.21514501,
        402.98205182, 402.98486847, 402.98768512, 402.99331845,
        402.99613514, 402.99895183, 407.10200156, 407.10483257,
        407.10766359, 407.21524937, 407.21808077, 407.22091218,
        416.0284147 , 416.03127658, 416.03413846, 417.03067114,
        417.03353647, 417.0364018 , 427.29065885, 427.29355921,
        427.29645957, 429.07332625, 429.07623266, 429.07913906,
        429.08204546, 429.0849519 , 429.08785833, 429.09076477,
        429.09367123, 429.09657769, 429.09948416, 429.10239064,
        429.10529714, 429.10820364, 429.11111014, 429.11982974,
        429.1227363 , 429.12564286, 430.07661627, 430.07952607,
        430.08243587, 430.08534567, 430.0882555 , 430.09116533,
        430.09407516, 430.09698502, 430.09989488, 430.10280474,
        430.10571462, 430.10862451, 431.07816473, 431.08107792,
        431.0839911 , 431.0869043 , 431.0898175 , 431.09273072,
        431.09564393, 431.09855718, 431.10147042, 432.10129416,
        432.1042108 , 432.10712744, 437.34662746, 437.34956175,
        437.35249604, 450.20087955, 450.20385665, 450.20683375,
        456.27663123, 456.27962835, 456.28262547, 459.28166587,
        459.28467285, 459.28767982, 459.95848086, 459.96149005,
        459.96449924, 469.35853575, 469.36157553, 469.36461531,
        475.02035599, 475.02341405, 475.02647211, 475.10904348,
        475.11210183, 475.11516017, 482.6993158 , 482.70239848,
        482.70548116, 489.05497683, 489.05807974, 489.06118264,
        491.05217203, 491.05528127, 491.05839051, 491.63687982,
        491.63999091, 491.643102  , 495.25859225, 495.26171478,
        495.2648373 , 495.26795983, 495.27108239, 495.27420494,
        495.2773275 , 495.28045008, 495.28357267, 503.10192504,
        503.10507219, 503.10821935, 503.11766086, 503.12080806,
        503.12395527, 503.12710247, 503.1302497 , 503.13339694,
        504.10321306, 504.10636335, 504.10951363, 504.67042105,
        504.67357311, 504.67672516, 505.11180331, 505.11495675,
        505.11811018, 505.7079754 , 505.71113069, 505.71428599,
        513.3248908 , 513.32806977, 513.33124874, 517.67971711,
        517.68290953, 517.68610196, 527.84588861, 527.84911223,
        527.85233585, 532.20348566, 532.20672256, 532.20995945,
        542.67400682, 542.6772754 , 542.68054398, 545.42307652,
        545.42635337, 545.42963022, 548.34335928, 548.34664489,
        548.3499305 , 551.32085548, 551.32415   , 551.32744452,
        551.33732813, 551.3406227 , 551.34391727, 552.33602997,
        552.33932753, 552.34262508, 552.34592263, 565.30464313,
        565.30797917, 565.3113152 , 574.49239018, 574.49575322,
        574.49911626, 575.37720624, 575.38057187, 575.3839375 ,
        583.32035694, 583.32374572, 583.3271345 , 586.36743749,
        586.37083511, 586.37423273, 601.20230697, 601.2057473 ,
        601.20918763, 605.59325757, 605.59671044, 605.60016331,
        607.36241018, 607.36586809, 607.369326  , 611.44611108,
        611.44958059, 611.45305011, 616.23294685, 616.23642992,
        616.23991299, 618.50948101, 618.5129705 , 618.51646   ,
        619.43104771, 619.43453981, 619.43803191, 632.48617379,
        632.48970249, 632.4932312 , 632.49675991, 632.50028864,
        633.51697091, 633.52050249, 633.52403407, 635.47850508,
        635.48204213, 635.48557917, 636.41262333, 636.41616297,
        636.41970261, 639.48517869, 639.48872686, 639.49227504,
        639.49582322, 639.49937142, 639.50291962, 639.50646784,
        639.51001606, 639.54195051, 639.54549884, 639.54904718,
        640.48615128, 640.48970223, 640.49325318, 640.5003551 ,
        640.50390609, 640.50745708, 640.99758813, 641.0011405 ,
        641.00469287, 641.51278288, 641.51633668, 641.51989047,
        642.51534048, 642.51889705, 642.52245362, 644.52636615,
        644.52992828, 644.53349041, 647.45418265, 647.45775287,
        647.46132308, 647.47560403, 647.4791743 , 647.48274457,
        651.412425  , 651.41600612, 651.41958723, 654.49580796,
        654.49939754, 654.50298711, 655.49767878, 655.5012711 ,
        655.50486343, 655.51204809, 655.51564045, 655.51923282,
        657.42816745, 657.43176506, 657.43536266, 658.50428803,
        658.50788859, 658.51148914, 661.48168964, 661.48529833,
        661.48890701, 662.05920248, 662.06281274, 662.066423  ,
        662.59001451, 662.59362622, 662.59723792, 663.43903339,
        663.44264741, 663.44626143, 663.45348948, 663.45710354,
        663.4607176 , 663.47155982, 663.47517392, 663.47878803,
        665.30155025, 665.30516933, 665.30878842, 665.51509264,
        665.51871231, 665.52233198, 666.74272062, 666.74634363,
        666.74996663, 673.24364226, 673.24728289, 673.25092351,
        675.50271216, 675.50635889, 675.51000561, 681.47091847,
        681.47458127, 681.47824407, 682.57386323, 682.577529  ,
        682.58119476, 684.18040583, 684.1840759 , 684.18774598,
        684.19141605, 684.19508616, 684.19875626, 684.20242638,
        684.2060965 , 684.20976664, 685.2010632 , 685.20473601,
        685.20840882, 688.55843003, 688.56211183, 688.56579363,
        689.43497357, 689.43865771, 689.44234185, 690.52221947,
        690.52590651, 690.52959356, 691.4553534 , 691.45904294,
        691.46273248, 692.55896037, 692.56265285, 692.56634533,
        695.33107742, 695.33477729, 695.33847715, 701.57929406,
        701.58301051, 701.58672696, 709.52136416, 709.52510159,
        709.52883901, 711.4811183 , 711.48486088, 711.48860347,
        712.50320612, 712.50695139, 712.51069666, 726.18223988,
        726.18602093, 726.18980198, 726.45071823, 726.45449998,
        726.45828173, 729.55126029, 729.55505011, 729.55883992,
        737.56202178, 737.56583234, 737.5696429 , 745.57438045,
        745.57821165, 745.58204286, 755.70759076, 755.71144791,
        755.71530506, 759.72046747, 759.72433485, 759.72820223,
        762.56951246, 762.57338709, 762.57726171, 767.6109324 ,
        767.61481981, 767.61870722, 768.68422807, 768.68811819,
        768.69200832, 773.52333774, 773.52724009, 773.53114244,
        773.55065433, 773.55455675, 773.55845917, 786.20677094,
        786.21070516, 786.21463937, 791.52691006, 791.53085756,
        791.53480506, 797.7009318 , 797.70489467, 797.70885754,
        800.75921765, 800.76318811, 800.76715857, 805.78572093,
        805.78970383, 805.79368673, 811.67518155, 811.67917898,
        811.68317641, 812.29489896, 812.29889791, 812.30289686,
        815.6294308 , 815.63343796, 815.63744511, 819.46877057,
        819.47278715, 819.47680372, 824.52536224, 824.52939119,
        824.53342013, 828.49055872, 828.49459735, 828.49863597,
        843.8308985 , 843.83497434, 843.83905019, 847.49904434,
        847.50312903, 847.50721372, 847.65018408, 847.65426914,
        847.65835419, 849.58759885, 849.59168857, 849.59577829,
        853.59203867, 853.59613802, 853.60023737, 868.38034383,
        868.38447854, 868.38861324, 871.31431326, 871.31845495,
        871.32259663, 874.67644411, 874.68059378, 874.68474345,
        883.86704228, 883.87121369, 883.8753851 , 883.87955652,
        883.88372795, 888.90908786, 888.91327115, 888.91745444,
        889.34838624, 889.35257057, 889.35675489, 889.69990309,
        889.70408824];

        let delta = [1.41051907e-03, 3.24446456e-02, 1.41076516e-03, 1.41076516e-03,
        5.95833186e+00, 1.45176300e-03, 1.45176300e-03, 2.03479401e+00,
        1.46551426e-03, 1.46551426e-03, 2.01404083e+00, 1.47899974e-03,
        1.47899974e-03, 1.47899974e-03, 1.47902927e-03, 1.47902927e-03,
        2.97925394e+00, 1.49874564e-03, 1.49874564e-03, 4.47942370e+00,
        1.52790185e-03, 1.52790185e-03, 2.25819780e+01, 1.66706811e-03,
        1.66706811e-03, 5.89164350e+00, 1.70152007e-03, 1.70152007e-03,
        1.70152007e-03, 1.70154960e-03, 1.70154960e-03, 4.59455503e-02,
        1.70183506e-03, 1.70183506e-03, 8.06366896e+00, 1.74787273e-03,
        1.74787273e-03, 2.39671019e+01, 1.87802242e-03, 1.87802242e-03,
        5.01895168e+00, 1.90416654e-03, 1.90416654e-03, 1.01441217e+00,
        1.90942293e-03, 1.90942293e-03, 3.90992651e+00, 1.92949366e-03,
        1.92949366e-03, 1.35067214e-02, 1.92958225e-03, 1.92958225e-03,
        3.92954563e+00, 1.94954471e-03, 1.94954471e-03, 6.01368644e+00,
        1.97969510e-03, 1.97969510e-03, 2.04829215e+00, 1.98987319e-03,
        1.98987319e-03, 1.03407245e+00, 1.99500161e-03, 1.99500161e-03,
        1.00873345e+00, 1.99999223e-03, 1.99999223e-03, 2.01496818e+00,
        2.00990455e-03, 2.00990455e-03, 1.83107442e+00, 2.01887191e-03,
        2.01887191e-03, 2.01888175e-03, 2.01889159e-03, 1.09543975e+01,
        2.07163263e-03, 2.07163263e-03, 3.05177032e+00, 2.08610246e-03,
        2.08610246e-03, 2.50339874e-02, 2.08624027e-03, 2.08624027e-03,
        2.08625011e-03, 2.08625995e-03, 2.08626980e-03, 2.08626980e-03,
        2.08629933e-03, 2.08629933e-03, 9.85832074e-01, 2.09096511e-03,
        2.09096511e-03, 9.71269708e-01, 2.09555214e-03, 2.09555214e-03,
        2.09556198e-03, 2.09557183e-03, 2.09558167e-03, 1.00179532e+01,
        2.14214104e-03, 2.14214104e-03, 3.14449381e+00, 2.15656165e-03,
        2.15656165e-03, 4.74682847e-01, 2.15874689e-03, 2.15874689e-03,
        2.78560876e-01, 2.16003638e-03, 2.16003638e-03, 2.21693778e+01,
        2.25882493e-03, 2.25882493e-03, 5.08903582e+00, 2.28091357e-03,
        2.28091357e-03, 9.12374286e-03, 2.28097263e-03, 2.28097263e-03,
        2.71445488e+00, 2.29267645e-03, 2.29267645e-03, 2.29267645e-03,
        4.58540212e-03, 2.29272567e-03, 2.29272567e-03, 9.95972100e-01,
        2.29701740e-03, 2.29701740e-03, 1.21419987e+00, 2.30223441e-03,
        2.30223441e-03, 6.90675245e-03, 2.30228363e-03, 2.30228363e-03,
        3.93739044e+00, 2.31907650e-03, 2.31907650e-03, 4.03842171e+00,
        2.33617452e-03, 2.33617452e-03, 2.00103131e+00, 2.34461033e-03,
        2.34461033e-03, 1.80591963e+00, 2.35219961e-03, 2.35219961e-03,
        2.35220945e-03, 2.35221929e-03, 2.35222914e-03, 2.35223898e-03,
        2.35224883e-03, 2.35225867e-03, 2.35226851e-03, 2.35226851e-03,
        2.35229804e-03, 2.35229804e-03, 2.07855992e+00, 2.36099962e-03,
        2.36099962e-03, 1.90379804e+00, 2.36894326e-03, 2.36894326e-03,
        1.61111225e-01, 2.36963230e-03, 2.36963230e-03, 2.00585767e+00,
        2.37796968e-03, 2.37796968e-03, 1.90241020e-02, 2.37806811e-03,
        2.37806811e-03, 7.95668382e+00, 2.41079748e-03, 2.41079748e-03,
        1.80399124e+01, 2.48338285e-03, 2.48338285e-03, 2.48338285e-03,
        2.48341239e-03, 2.48341239e-03, 7.45028637e-03, 2.48346160e-03,
        2.48346160e-03, 2.48347145e-03, 1.73845659e-02, 2.48356004e-03,
        2.48356004e-03, 6.99005695e+00, 2.51113145e-03, 2.51113145e-03,
        2.95990297e+00, 2.52272700e-03, 2.52272700e-03, 3.92459687e+00,
        2.53801383e-03, 2.53801383e-03, 9.08485734e+00, 2.57302687e-03,
        2.57302687e-03, 3.00426190e+00, 2.58451414e-03, 2.58451414e-03,
        6.87830302e-01, 2.58715218e-03, 2.58715218e-03, 1.01233085e+00,
        2.59102064e-03, 2.59102064e-03, 2.59103049e-03, 2.59104033e-03,
        2.59105017e-03, 2.59106002e-03, 2.59106986e-03, 2.59107970e-03,
        2.59108955e-03, 2.59109939e-03, 2.59110923e-03, 2.59111908e-03,
        1.14018975e-01, 2.59157187e-03, 2.59157187e-03, 8.50566677e-01,
        2.59482020e-03, 2.59482020e-03, 2.59483005e-03, 2.59483989e-03,
        2.59483989e-03, 2.59486942e-03, 2.59486942e-03, 9.81565725e-01,
        2.59860992e-03, 2.59860992e-03, 2.59861976e-03, 2.59862961e-03,
        7.79593804e-03, 2.59867882e-03, 2.59867882e-03, 2.59868867e-03,
        2.72917079e-01, 2.59974191e-03, 2.59974191e-03, 1.14398379e-01,
        2.60019471e-03, 2.60019471e-03, 5.90498917e-01, 2.60244885e-03,
        2.60244885e-03, 9.79218421e-01, 2.60616967e-03, 2.60616967e-03,
        1.00512009e+01, 2.64387980e-03, 2.64387980e-03, 2.64388965e-03,
        2.64388965e-03, 7.93175753e-03, 2.64394871e-03, 2.64394871e-03,
        2.64394871e-03, 2.64396839e-03, 2.64398808e-03, 2.64398808e-03,
        1.90393001e-01, 2.64471649e-03, 2.64471649e-03, 7.75325889e-01,
        2.64762030e-03, 2.64762030e-03, 7.94291013e-03, 2.64766952e-03,
        2.64766952e-03, 9.75010702e-01, 2.65131159e-03, 2.65131159e-03,
        5.30264286e-03, 2.65135096e-03, 2.65135096e-03, 2.59880135e-01,
        2.65233530e-03, 2.65233530e-03, 7.13835650e-01, 2.65500287e-03,
        2.65500287e-03, 9.61757770e-01, 2.65858587e-03, 2.65858587e-03,
        2.39277060e-02, 2.65869415e-03, 2.65869415e-03, 2.15386906e-01,
        2.65951115e-03, 2.65951115e-03, 1.98940486e+00, 2.66688387e-03,
        2.66688387e-03, 7.79151163e-01, 2.66977784e-03, 2.66977784e-03,
        7.06365868e+00, 2.69571525e-03, 2.69571525e-03, 1.23297200e+00,
        2.70023338e-03, 2.70023338e-03, 7.42937727e-01, 2.70296001e-03,
        2.70296001e-03, 5.40593970e-03, 2.70299938e-03, 2.70299938e-03,
        9.76425945e-01, 2.70657254e-03, 2.70657254e-03, 1.35330005e-02,
        2.70664144e-03, 2.70664144e-03, 2.96696159e+00, 2.71742983e-03,
        2.71742983e-03, 1.01797199e+01, 2.75407688e-03, 2.75407688e-03,
        2.75407688e-03, 1.09690809e+00, 2.75802409e-03, 2.75802409e-03,
        4.79722068e+00, 2.77511227e-03, 2.77511227e-03, 1.18865173e+00,
        2.77934494e-03, 2.77934494e-03, 9.26069257e-01, 2.78264248e-03,
        2.78264248e-03, 7.67826199e+00, 2.80969220e-03, 2.80969220e-03,
        2.10754959e-01, 2.81045014e-03, 2.81045014e-03, 1.76690681e+00,
        2.81665150e-03, 2.81665150e-03, 5.63332268e-03, 2.81669087e-03,
        2.81669087e-03, 4.10304973e+00, 2.83101304e-03, 2.83101304e-03,
        1.07585780e-01, 2.83140678e-03, 2.83140678e-03, 8.80750252e+00,
        2.86188201e-03, 2.86188201e-03, 9.96532680e-01, 2.86532720e-03,
        2.86532720e-03, 1.02542571e+01, 2.90035993e-03, 2.90035993e-03,
        1.77686668e+00, 2.90640379e-03, 2.90640379e-03, 2.90640379e-03,
        2.90643332e-03, 2.90643332e-03, 2.90644317e-03, 2.90645301e-03,
        2.90646285e-03, 2.90647270e-03, 2.90648254e-03, 2.90649238e-03,
        2.90650223e-03, 2.90650223e-03, 8.71959527e-03, 2.90656129e-03,
        2.90656129e-03, 9.50973414e-01, 2.90979977e-03, 2.90979977e-03,
        2.90979977e-03, 2.90982930e-03, 2.90982930e-03, 2.90982930e-03,
        2.90985883e-03, 2.90985883e-03, 2.90985883e-03, 2.90988836e-03,
        2.90988836e-03, 9.69540217e-01, 2.91318591e-03, 2.91318591e-03,
        2.91319575e-03, 2.91320559e-03, 2.91321544e-03, 2.91321544e-03,
        2.91324497e-03, 2.91324497e-03, 9.99823737e-01, 2.91664095e-03,
        2.91664095e-03, 5.23950002e+00, 2.93429020e-03, 2.93429020e-03,
        1.28483835e+01, 2.97709923e-03, 2.97709923e-03, 6.06979748e+00,
        2.99712074e-03, 2.99712074e-03, 2.99904040e+00, 3.00697401e-03,
        3.00697401e-03, 6.70801043e-01, 3.00918877e-03, 3.00918877e-03,
        9.39403650e+00, 3.03978212e-03, 3.03978212e-03, 5.65574068e+00,
        3.05806135e-03, 3.05806135e-03, 8.25713674e-02, 3.05834681e-03,
        3.05834681e-03, 7.58415562e+00, 3.08267974e-03, 3.08267974e-03,
        6.34949567e+00, 3.10290797e-03, 3.10290797e-03, 1.99098939e+00,
        3.10923729e-03, 3.10923729e-03, 5.78489312e-01, 3.11108785e-03,
        3.11108785e-03, 3.61549026e+00, 3.12252590e-03, 3.12252590e-03,
        3.12252590e-03, 3.12255543e-03, 3.12255543e-03, 3.12255543e-03,
        3.12258496e-03, 3.12258496e-03, 7.81835237e+00, 3.14715414e-03,
        3.14715414e-03, 9.44151163e-03, 3.14720335e-03, 3.14720335e-03,
        3.14720335e-03, 3.14723288e-03, 3.14723288e-03, 9.69816127e-01,
        3.15028434e-03, 3.15028434e-03, 5.60907419e-01, 3.15205616e-03,
        3.15205616e-03, 4.35078148e-01, 3.15343424e-03, 3.15343424e-03,
        5.89865220e-01, 3.15529464e-03, 3.15529464e-03, 7.61060481e+00,
        3.17896807e-03, 3.17896807e-03, 4.34846837e+00, 3.19242402e-03,
        3.19242402e-03, 1.01597867e+01, 3.22361782e-03, 3.22361782e-03,
        4.35114981e+00, 3.23689659e-03, 3.23689659e-03, 1.04640474e+01,
        3.26858256e-03, 3.26858256e-03, 2.74253254e+00, 3.27685103e-03,
        3.27685103e-03, 2.91372906e+00, 3.28561167e-03, 3.28561167e-03,
        2.97092498e+00, 3.29451996e-03, 3.29451996e-03, 9.88360911e-03,
        3.29456918e-03, 3.29456918e-03, 9.92112707e-01, 3.29755174e-03,
        3.29755174e-03, 3.29755174e-03, 1.29587205e+01, 3.33603951e-03,
        3.33603951e-03, 9.18107497e+00, 3.36304001e-03, 3.36304001e-03,
        8.78089988e-01, 3.36562882e-03, 3.36562882e-03, 7.93641944e+00,
        3.38878055e-03, 3.38878055e-03, 3.04030300e+00, 3.39761994e-03,
        3.39761994e-03, 1.48280742e+01, 3.44033053e-03, 3.44033053e-03,
        4.38406994e+00, 3.45287105e-03, 3.45287105e-03, 1.76224687e+00,
        3.45791088e-03, 3.45791088e-03, 4.07678507e+00, 3.46951627e-03,
        3.46951627e-03, 4.77989674e+00, 3.48307066e-03, 3.48307066e-03,
        2.26956802e+00, 3.48949841e-03, 3.48949841e-03, 9.14587709e-01,
        3.49209707e-03, 3.49209707e-03, 1.30481419e+01, 3.52870475e-03,
        3.52870475e-03, 3.52871459e-03, 3.52872444e-03, 1.01668227e+00,
        3.53157903e-03, 3.53157903e-03, 1.95447102e+00, 3.53704213e-03,
        3.53704213e-03, 9.27044162e-01, 3.53964079e-03, 3.53964079e-03,
        3.06547608e+00, 3.54817503e-03, 3.54817503e-03, 3.54818488e-03,
        3.54819472e-03, 3.54820456e-03, 3.54821441e-03, 3.54822425e-03,
        3.19344514e-02, 3.54833253e-03, 3.54833253e-03, 9.37104100e-01,
        3.55095088e-03, 3.55095088e-03, 7.10192144e-03, 3.55099025e-03,
        3.55099025e-03, 4.90131053e-01, 3.55236833e-03, 3.55236833e-03,
        5.08090009e-01, 3.55379562e-03, 3.55379562e-03, 9.95450005e-01,
        3.55657147e-03, 3.55657147e-03, 2.00391253e+00, 3.56213300e-03,
        3.56213300e-03, 2.92069224e+00, 3.57021445e-03, 3.57021445e-03,
        1.42809464e-02, 3.57027351e-03, 3.57027351e-03, 3.92968043e+00,
        3.58111111e-03, 3.58111111e-03, 3.07622073e+00, 3.58957645e-03,
        3.58957645e-03, 9.94691669e-01, 3.59232277e-03, 3.59232277e-03,
        7.18466522e-03, 3.59236214e-03, 3.59236214e-03, 1.90893463e+00,
        3.59760868e-03, 3.59760868e-03, 1.06892537e+00, 3.60055187e-03,
        3.60055187e-03, 2.97020051e+00, 3.60868253e-03, 3.60868253e-03,
        5.70295473e-01, 3.61025748e-03, 3.61025748e-03, 5.23591517e-01,
        3.61170446e-03, 3.61170446e-03, 8.41795471e-01, 3.61401766e-03,
        3.61401766e-03, 7.22805501e-03, 3.61405704e-03, 3.61405704e-03,
        1.08422203e-02, 3.61410625e-03, 3.61410625e-03, 1.82276222e+00,
        3.61908702e-03, 3.61908702e-03, 2.06304222e-01, 3.61966779e-03,
        3.61966779e-03, 1.22038865e+00, 3.62300470e-03, 3.62300470e-03,
        6.49367563e+00, 3.64062442e-03, 3.64062442e-03, 2.25178865e+00,
        3.64672734e-03, 3.64672734e-03, 5.96091286e+00, 3.66280165e-03,
        3.66280165e-03, 1.09561916e+00, 3.66576452e-03, 3.66576452e-03,
        1.59921106e+00, 3.67007593e-03, 3.67007593e-03, 3.67007593e-03,
        3.67010547e-03, 3.67010547e-03, 3.67011531e-03, 3.67012515e-03,
        3.67013500e-03, 9.91296560e-01, 3.67281241e-03, 3.67281241e-03,
        3.35002121e+00, 3.68179945e-03, 3.68179945e-03, 8.69179941e-01,
        3.68414218e-03, 3.68414218e-03, 1.07987762e+00, 3.68704599e-03,
        3.68704599e-03, 9.25759841e-01, 3.68953637e-03, 3.68953637e-03,
        1.09622789e+00, 3.69247956e-03, 3.69247956e-03, 2.76473210e+00,
        3.69986212e-03, 3.69986212e-03, 6.24081691e+00, 3.71644828e-03,
        3.71644828e-03, 7.93463720e+00, 3.73742461e-03, 3.73742461e-03,
        1.95227929e+00, 3.74258256e-03, 3.74258256e-03, 1.01460265e+00,
        3.74526981e-03, 3.74526981e-03, 1.36715432e+01, 3.78105064e-03,
        3.78105064e-03, 2.60916256e-01, 3.78174952e-03, 3.78174952e-03,
        3.09297856e+00, 3.78981129e-03, 3.78981129e-03, 8.00318187e+00,
        3.81056121e-03, 3.81056121e-03, 8.00473755e+00, 3.83120286e-03,
        3.83120286e-03, 1.01255479e+01, 3.85715012e-03, 3.85715012e-03,
        4.00516242e+00, 3.86737743e-03, 3.86737743e-03, 2.84131024e+00,
        3.87462218e-03, 3.87462218e-03, 5.03367069e+00, 3.88740879e-03,
        3.88740879e-03, 1.06552085e+00, 3.89012557e-03, 3.89012557e-03,
        4.83132942e+00, 3.90235110e-03, 3.90235110e-03, 1.95118933e-02,
        3.90242000e-03, 3.90242000e-03, 1.26483118e+01, 3.93421424e-03,
        3.93421424e-03, 5.31227068e+00, 3.94750286e-03, 3.94750286e-03,
        6.16612674e+00, 3.96286844e-03, 3.96286844e-03, 3.05036011e+00,
        3.97045771e-03, 3.97045771e-03, 5.01856237e+00, 3.98289979e-03,
        3.98289979e-03, 5.88149482e+00, 3.99742868e-03, 3.99742868e-03,
        6.11722544e-01, 3.99895441e-03, 3.99895441e-03, 3.32653394e+00,
        4.00715398e-03, 4.00715398e-03, 3.83132546e+00, 4.01657413e-03,
        4.01657413e-03, 5.04855852e+00, 4.02894731e-03, 4.02894731e-03,
        3.95713859e+00, 4.03862339e-03, 4.03862339e-03, 1.53322625e+01,
        4.07584136e-03, 4.07584136e-03, 3.65999416e+00, 4.08469059e-03,
        4.08469059e-03, 1.42970362e-01, 4.08505480e-03, 4.08505480e-03,
        1.92924466e+00, 4.08972058e-03, 4.08972058e-03, 3.99626038e+00,
        4.09934744e-03, 4.09934744e-03, 1.47801065e+01, 4.13470500e-03,
        4.13470500e-03, 2.92570002e+00, 4.14168399e-03, 4.14168399e-03,
        3.35384748e+00, 4.14966700e-03, 4.14966700e-03, 9.18229883e+00,
        4.17141111e-03, 4.17141111e-03, 4.17142096e-03, 4.17143080e-03,
        5.02535991e+00, 4.18329212e-03, 4.18329212e-03, 4.30931800e-01,
        4.18432568e-03, 4.18432568e-03, 3.43148193e-01, 4.18515253e-03];

        let weights = [11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,
        0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0., 13.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 13., 13., 13.,  0.,  0., 11.,
        0.,  0., 13.,  0.,  0., 40., 13., 40., 26.,  0.,  0., 13.,  0.,
        0., 11.,  0.,  0., 13., 13.,  0., 13.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,
        0.,  0., 13.,  0.,  0., 13., 26.,  0.,  0., 13.,  0.,  0., 11.,
        0.,  0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 13.,  0., 13., 26., 26., 13., 13., 26.,  0.,
        0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 13., 13.,  0., 13.,
        0.,  0., 13., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 26., 26.,  0., 66., 26., 40., 26., 13., 13.,  0., 13.,  0.,
        0., 11.,  0.,  0., 13., 13., 13.,  0.,  0., 40.,  0.,  0., 13.,
        0., 13.,  0.,  0., 13., 26.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 26., 13., 26.,  0.,  0., 13.,
       13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0., 26.,
        0.,  0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0., 13.,
        0.,  0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 13., 13.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 13., 26.,  0., 13.,  0., 40., 13., 40.,  0., 26., 26.,
       13.,  0.,  0., 13.,  0.,  0., 13., 13.,  0., 13.,  0.,  0., 26.,
        0.,  0., 13.,  0.,  0., 13., 26.,  0., 13.,  0.,  0., 13.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 13.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0.,
       13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       13.,  0.,  0., 13.,  0.,  0., 13., 13.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0., 13.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13., 13.,  0., 13.,  0.,
       13.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,
        0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0.,
       13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0., 13.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 13.,  0.,  0., 13.,
       13.,  0., 26.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 13.,  0.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0.,
       11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,
        0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.,  0.,
        0., 13.,  0., 13.,  0.,  0., 11.,  0.,  0., 11.,  0.,  0., 11.];

        let weights_trans: Vec<_> = weights.iter().map(|v| (v + 1.0).ln().sqrt()).collect();

        let m = select_delta_model(&data, Some(&weights_trans));
        eprintln!("Selected betas: {m:?}");

        let (delta, data, weights): (Vec<_>, Vec<_>, Vec<_>) = delta.into_iter().zip(data).zip(weights).filter_map(|((delta, mz), weight)| {
            if delta >= 1.0 {
                None
            } else {
                Some((delta, mz, (weight + 1.0).ln().sqrt()))
            }
        }).collect();

        let betas = fit_delta_model(&data, &delta, Some(&weights), 2).unwrap();
        eprintln!("{betas:?}");
    }
}
