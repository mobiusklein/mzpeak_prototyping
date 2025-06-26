use arrow::{
    array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch, UInt64Array},
    compute::{take_arrays, take_record_batch},
};

use num_traits::Float;


fn _find_zeros<T: Float, I: Iterator<Item = Option<T>>>(iter: I) -> Vec<u64> {
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


pub fn find_zeros(array: &impl Array) -> Option<Vec<u64>> {
    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        Some(_find_zeros(array.iter()))
    } else if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        Some(_find_zeros(array.iter()))
    } else {
        None
    }
}

pub fn drop_where_column_is_zero(batch: &RecordBatch, column_index: usize) -> Result<RecordBatch, arrow::error::ArrowError> {
    let target_array = batch.column(column_index);
    if let Some(indices) = find_zeros(target_array) {
        take_record_batch(&batch, &UInt64Array::from(indices))
    } else {
        Ok(batch.clone())
    }
}

pub fn drop_where_column_is_zero_arrays(arrays: &[ArrayRef], column_index: usize) -> Result<Vec<std::sync::Arc<dyn Array + 'static>>, arrow::error::ArrowError> {
    let target_array = &arrays[column_index];
    if let Some(indices) = find_zeros(target_array) {
        take_arrays(&arrays, &UInt64Array::from(indices), None)
    } else {
        Ok(arrays.to_vec())
    }
}
