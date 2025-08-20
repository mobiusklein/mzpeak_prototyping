#![allow(unused)]
use std::collections::{HashMap, HashSet};
use std::ops::AddAssign;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, ArrowPrimitiveType, AsArray, Float32Array, Float32Builder,
    Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array, Int64Builder,
    LargeBinaryArray, LargeBinaryBuilder, LargeListBuilder, PrimitiveArray, StructArray,
    StructBuilder, UInt8Array, UInt8Builder, UInt32Builder, UInt64Array, UInt64Builder,
    new_null_array,
};
use arrow::compute::kernels::nullif;
use arrow::datatypes::{
    DataType, Field, FieldRef, Fields, Float32Type, Float64Type, Int32Type, Int64Type, Schema,
    UInt8Type,
};
use itertools::Itertools;
use mzdata::params::Unit;
use mzdata::spectrum::bindata::BinaryCompressionType;
use mzdata::{
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, BinaryDataArrayType, DataArray,
        bindata::{ArrayRetrievalError, DataArraySlice},
    },
};

use mzpeaks::coordinate::SimpleInterval;
use serde::{Deserialize, Serialize};

use bytemuck::Pod;
use num_traits::{Float, NumCast, ToPrimitive, Zero};

use crate::filter::{
    _skip_zero_runs_gen, _skip_zero_runs_iter, MZDeltaModel, RegressionDeltaModel, fill_nulls_for,
    is_zero_pair_mask, null_chunk_every_k, null_delta_decode, null_delta_encode, take_data_array,
};
use crate::peak_series::{
    BufferContext, BufferFormat, BufferName, MZ_ARRAY, array_to_arrow_type,
    data_array_to_arrow_array,
};
use crate::spectrum::AuxiliaryArray;
use crate::{CURIE, curie};

pub fn chunk_every_k<T: Float>(data: &[T], k: T) -> Vec<SimpleInterval<usize>> {
    let mut chunks = Vec::new();
    if data.is_empty() {
        return chunks;
    }
    let start = data[0];
    let end = data[data.len() - 1];

    let mut t = start + k;
    if t > end {
        chunks.push(SimpleInterval::new(0, data.len()));
        return chunks;
    }
    let mut offset = 0;
    for (i, v) in data.iter().copied().enumerate() {
        if v > t {
            chunks.push(SimpleInterval::new(offset, i));
            offset = i;
            while t < v {
                t = t + k;
            }
        }
    }
    chunks.push(SimpleInterval::new(offset, data.len()));
    chunks
}

pub fn delta_encode<T: Float + Pod>(
    data: &[T],
    name: &ArrayType,
    dtype: &BinaryDataArrayType,
) -> (f64, f64, DataArray) {
    let start = data[0].to_f64().unwrap();
    let end = data.last().copied().unwrap().to_f64().unwrap();
    let mut acc =
        DataArray::from_name_type_size(name, *dtype, data.len() * core::mem::size_of::<T>());

    let mut last = data[0];
    for v in data.iter().copied().skip(1) {
        let delta = v - last;
        last = v;
        acc.push(delta).unwrap();
    }

    (start, end, acc)
}

pub fn delta_decode<T: Float + Pod + AddAssign>(
    it: &[T],
    start_value: T,
    accumulator: &mut DataArray,
) {
    let mut state = start_value;
    accumulator.push(state);
    for val in it.iter().copied() {
        state += val;
        accumulator.push(state).unwrap();
    }
}

pub const NO_COMPRESSION: CURIE = curie!(MS:1000576);
pub const DELTA_ENCODE: CURIE = curie!(MS:1003089);
pub const NUMPRESS_LINEAR: CURIE = curie!(MS:1002312);
pub const NUMPRESS_SLOF: CURIE = curie!(MS:1002314);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkingStrategy {
    Basic { chunk_size: f64 },
    Delta { chunk_size: f64 },
    NumpressLinear { chunk_size: f64 },
}

impl ChunkingStrategy {
    pub const fn as_curie(&self) -> CURIE {
        match self {
            Self::Basic { chunk_size } => NO_COMPRESSION,
            Self::Delta { chunk_size } => DELTA_ENCODE,
            Self::NumpressLinear { chunk_size } => NUMPRESS_LINEAR,
        }
    }

    pub fn extra_arrays(&self, main_axis_name: &BufferName) -> Vec<Field> {
        match self {
            ChunkingStrategy::Basic { chunk_size } => vec![],
            ChunkingStrategy::Delta { chunk_size } => vec![],
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                let meta = BufferName::new(
                    main_axis_name.context,
                    ArrayType::nonstandard(format!("{}_numpress_bytes", main_axis_name)),
                    main_axis_name.dtype,
                )
                .with_format(BufferFormat::Chunked)
                .with_transform(Some(curie!(MS:1002312)))
                .as_field_metadata();
                let bytes = Field::new(
                    format!("{}_numpress_bytes", main_axis_name),
                    DataType::LargeList(Arc::new(Field::new("item", DataType::UInt8, false))),
                    true,
                )
                .with_metadata(meta);
                vec![bytes]
            }
        }
    }

    pub fn encode_extra_arrow(
        &self,
        main_axis_name: &BufferName,
        chunk: &ArrowArrayChunk,
        chunk_builder: &mut StructBuilder,
        schema: &Schema,
        visited: &mut HashSet<usize>,
    ) {
        match self {
            ChunkingStrategy::Basic { chunk_size } => {}
            ChunkingStrategy::Delta { chunk_size } => {}
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                let fields = self.extra_arrays(main_axis_name);
                let byte_col = &fields[0];
                let idx = schema
                    .fields()
                    .iter()
                    .position(|p| p.name() == byte_col.name())
                    .unwrap();

                if visited.contains(&idx) {
                    return;
                }
                visited.insert(idx);

                let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                    chunk_builder.field_builder(idx).unwrap();
                let inner = b
                    .values()
                    .as_any_mut()
                    .downcast_mut::<UInt8Builder>()
                    .unwrap();
                if matches!(chunk.chunk_encoding, Self::NumpressLinear { chunk_size: _ }) {
                    let bytes: &UInt8Array = chunk.chunk_values.as_primitive();
                    inner.extend(bytes);
                    b.append(true);
                } else {
                    b.append_null();
                }
            }
        }
    }

    pub const fn chunk_size(&self) -> f64 {
        match self {
            ChunkingStrategy::Basic { chunk_size } => *chunk_size,
            ChunkingStrategy::Delta { chunk_size } => *chunk_size,
            ChunkingStrategy::NumpressLinear { chunk_size } => *chunk_size,
        }
    }

    pub fn encode<T: Float + Pod>(
        &self,
        data: &[T],
        name: &ArrayType,
        dtype: &BinaryDataArrayType,
    ) -> (f64, f64, DataArray) {
        match self {
            ChunkingStrategy::Basic { chunk_size } => {
                let mut acc =
                    DataArray::from_name_type_size(name, *dtype, dtype.size_of() * data.len());
                acc.extend(data).unwrap();
                let start = data.first().copied().unwrap_or_else(|| T::zero());
                let end = data.last().copied().unwrap_or_else(|| T::zero());
                (start.to_f64().unwrap(), end.to_f64().unwrap(), acc)
            }
            ChunkingStrategy::Delta { chunk_size } => delta_encode(data, name, dtype),
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                let mut acc =
                    DataArray::from_name_type_size(name, *dtype, dtype.size_of() * data.len());
                acc.extend(data).unwrap();
                acc.store_compressed(BinaryCompressionType::NumpressLinear);
                let start = data.first().copied().unwrap_or_else(|| T::zero());
                let end = data.last().copied().unwrap_or_else(|| T::zero());
                (start.to_f64().unwrap(), end.to_f64().unwrap(), acc)
            }
        }
    }

    pub fn encode_arrow<T: ArrowPrimitiveType>(
        &self,
        array: &PrimitiveArray<T>,
    ) -> (f64, f64, ArrayRef)
    where
        T::Native: Float,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        // Need to find the first non-null value
        let mut it = array.iter().filter(|v| v.is_some());
        let start: f64 = it
            .next()
            .and_then(|v| v.map(|v| v.to_f64().unwrap_or(0.0)))
            .unwrap_or(0.0);
        let end: f64 = it
            .next_back()
            .and_then(|v| v.map(|v| v.to_f64().unwrap_or(0.0)))
            .unwrap_or(0.0);
        match self {
            ChunkingStrategy::Basic { chunk_size } => (start, end, Arc::new(array.clone())),
            ChunkingStrategy::Delta { chunk_size } => {
                (start, end, Arc::new(null_delta_encode(array)))
            }
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                // if array.null_count() > 0 {
                //     log::warn!(
                //         "Numpress compression on an array with null values is poorly supported at read time."
                //     );
                // }
                let values: Vec<_> = array
                    .iter()
                    .map(|v| v.and_then(|v| v.to_f64()).unwrap_or_default())
                    .collect();
                let bytes_of = DataArray::compress_numpress_linear(&values).unwrap();
                let array = Arc::new(UInt8Array::from(bytes_of));
                (start, end, array)
            }
        }
    }

    pub fn decode_arrow(
        &self,
        array: &ArrayRef,
        start_value: f64,
        accumulator: &mut DataArray,
        delta_model: Option<&RegressionDeltaModel<f64>>,
    ) {
        match self {
            ChunkingStrategy::Basic { chunk_size } => match array.data_type() {
                DataType::Float32 => {
                    let it = array.as_primitive::<Float32Type>();
                    accumulator.extend(it.values()).unwrap();
                }
                DataType::Float64 => {
                    let it = array.as_primitive::<Float64Type>();
                    accumulator.extend(it.values()).unwrap();
                }
                _ => panic!(
                    "Data type {:?} is not supported by basic decoding",
                    array.data_type()
                ),
            },
            ChunkingStrategy::Delta { chunk_size } => match array.data_type() {
                DataType::Float32 => {
                    let it = array.as_primitive::<Float32Type>();
                    if it.null_count() > 0 {
                        let decoded = null_delta_decode(it, start_value as f32);
                        if let Some(delta_model) = delta_model {
                            let values = fill_nulls_for(&decoded, delta_model);
                            accumulator.extend(&values).unwrap();
                        } else {
                            log::debug!(
                                "f32 delta decoding contained nulls but no delta model provided"
                            );
                            accumulator.extend(decoded.values()).unwrap()
                        }
                    } else {
                        delta_decode(it.values(), start_value as f32, accumulator);
                    }
                }
                DataType::Float64 => {
                    let it = array.as_primitive::<Float64Type>();
                    if it.null_count() > 0 {
                        let decoded = null_delta_decode(it, start_value);
                        if let Some(delta_model) = delta_model {
                            let values = fill_nulls_for(&decoded, delta_model);
                            accumulator.extend(&values).unwrap();
                        } else {
                            log::debug!(
                                "f64 delta decoding contained nulls but no delta model provided"
                            );
                            accumulator.extend(decoded.values()).unwrap()
                        }
                    } else {
                        delta_decode(it.values(), start_value, accumulator);
                    }
                }
                _ => panic!(
                    "Data type {:?} is not supported by chunk decoding",
                    array.data_type()
                ),
            },
            ChunkingStrategy::NumpressLinear { chunk_size } => match array.data_type() {
                DataType::UInt8 => {
                    let it = array.as_primitive::<UInt8Type>();
                    let buf = it.values();
                    let data: Float64Array = DataArray::decompress_numpress_linear(buf)
                        .unwrap()
                        .into_iter()
                        .map(|v| if v == 0.0 { None } else { Some(v) })
                        .collect();
                    if let Some(delta_model) = delta_model {
                        if data.null_count() > 0 {
                            let data = fill_nulls_for(&data, delta_model);
                            match accumulator.dtype() {
                                BinaryDataArrayType::Float64 => {
                                    accumulator.extend(&data).unwrap();
                                }
                                BinaryDataArrayType::Float32 => {
                                    for v in data {
                                        accumulator.push(v as f32).unwrap();
                                    }
                                }
                                _ => unimplemented!(),
                            }
                        }
                    }
                }
                _ => panic!(
                    "Data type {:?} is not supported by numpress linear decoding",
                    array.data_type()
                ),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrowArrayChunk {
    /// The index of source entity
    pub series_index: u64,
    /// The starting coordinate of the chunk axis
    pub chunk_start: f64,
    /// The ending coordinate of the chunk axis
    pub chunk_end: f64,
    /// The buffer name for the main axis of the chunk
    pub chunk_axis: BufferName,
    /// The array values of the chunk, encoded using [`Self::chunk_values`] as [`ChunkingStrategy`]
    pub chunk_values: ArrayRef,
    /// The chunk encoding strategy applied to [`Self::chunk_values`].
    pub chunk_encoding: ChunkingStrategy,
    /// The rest of the arrays of covering this chunk
    pub arrays: HashMap<BufferName, ArrayRef>,
}

impl ArrowArrayChunk {
    pub fn new(
        series_index: u64,
        chunk_start: f64,
        chunk_end: f64,
        chunk_axis: BufferName,
        chunk_values: ArrayRef,
        chunk_encoding: ChunkingStrategy,
        arrays: HashMap<BufferName, ArrayRef>,
    ) -> Self {
        Self {
            series_index,
            chunk_start,
            chunk_end,
            chunk_axis,
            chunk_values,
            chunk_encoding,
            arrays,
        }
    }

    /// Convert a series of [`ArrowArrayChunk`] into a [`StructArray`]
    pub fn to_struct_array(
        chunks: &[Self],
        series_index_name: impl Into<String>,
        schema: &Fields,
        encodings: &[ChunkingStrategy],
    ) -> StructArray {
        let this_schema = chunks[0].to_schema(series_index_name, encodings);
        let mut this_builder =
            StructBuilder::from_fields(this_schema.fields().clone(), chunks.len());

        let mut visited: HashSet<usize> = HashSet::new();
        for chunk in chunks {
            visited.clear();
            let mut b = this_builder.field_builder::<UInt64Builder>(0).unwrap();
            b.append_value(chunk.series_index);
            let mut b = this_builder.field_builder::<Float64Builder>(1).unwrap();
            b.append_value(chunk.chunk_start);
            let mut b = this_builder.field_builder::<Float64Builder>(2).unwrap();
            b.append_value(chunk.chunk_end);

            let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                this_builder.field_builder(3).unwrap();
            if matches!(
                chunk.chunk_encoding,
                ChunkingStrategy::NumpressLinear { chunk_size }
            ) {
                b.append_null();
            } else {
                match array_to_arrow_type(chunk.chunk_axis.dtype) {
                    DataType::Int32 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Int32Builder>()
                            .unwrap();
                        inner.append_array(chunk.chunk_values.as_primitive());
                    }
                    DataType::Int64 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Int64Builder>()
                            .unwrap();
                        inner.append_array(chunk.chunk_values.as_primitive());
                    }
                    DataType::Float32 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Float32Builder>()
                            .unwrap();
                        inner.append_array(chunk.chunk_values.as_primitive());
                    }
                    DataType::Float64 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Float64Builder>()
                            .unwrap();
                        inner.append_array(chunk.chunk_values.as_primitive());
                    }
                    DataType::LargeBinary => todo!(),
                    tp => {
                        unimplemented!(
                            "Array type {tp:?} from {:?} not supported",
                            chunk.chunk_axis.dtype
                        )
                    }
                }
                b.append(true);
            }

            for encoding in encodings {
                encoding.encode_extra_arrow(
                    &chunk.chunk_axis,
                    &chunk,
                    &mut this_builder,
                    &this_schema,
                    &mut visited,
                );
            }

            let mut cb = this_builder.field_builder::<StructBuilder>(4).unwrap();
            let curie_of = chunk.chunk_encoding.as_curie();
            cb.field_builder::<UInt8Builder>(0)
                .unwrap()
                .append_value(curie_of.cv_id);
            cb.field_builder::<UInt32Builder>(1)
                .unwrap()
                .append_value(curie_of.accession);
            cb.append(true);

            visited.insert(0);
            visited.insert(1);
            visited.insert(2);
            visited.insert(3);
            visited.insert(4);

            for (i, f) in this_schema.fields().iter().enumerate().skip(5) {
                if visited.contains(&i) {
                    continue;
                }
                if let Some(buf_name) = BufferName::from_field(chunk.chunk_axis.context, f.clone())
                    .map(|f| f.with_format(BufferFormat::ChunkedSecondary))
                {
                    let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                        this_builder.field_builder(i).unwrap();
                    if let Some(arr) = chunk.arrays.get(&buf_name) {
                        match array_to_arrow_type(buf_name.dtype) {
                            DataType::Int32 => {
                                let inner = b
                                    .values()
                                    .as_any_mut()
                                    .downcast_mut::<Int32Builder>()
                                    .unwrap();
                                inner.append_array(arr.as_primitive());
                            }
                            DataType::Int64 => {
                                let inner = b
                                    .values()
                                    .as_any_mut()
                                    .downcast_mut::<Int64Builder>()
                                    .unwrap();
                                inner.append_array(arr.as_primitive());
                            }
                            DataType::Float32 => {
                                let inner = b
                                    .values()
                                    .as_any_mut()
                                    .downcast_mut::<Float32Builder>()
                                    .unwrap();
                                inner.append_array(arr.as_primitive());
                            }
                            DataType::Float64 => {
                                let inner = b
                                    .values()
                                    .as_any_mut()
                                    .downcast_mut::<Float64Builder>()
                                    .unwrap();
                                inner.append_array(arr.as_primitive());
                            }
                            DataType::LargeBinary => todo!(),
                            tp => {
                                unimplemented!(
                                    "Array type {tp:?} from {:?} not supported",
                                    buf_name.dtype
                                )
                            }
                        }
                        b.append(true);
                    } else {
                        b.append_null();
                    }
                } else {
                    if !visited.contains(&i) {
                        panic!("A column was not visited: {}", f.name());
                    }
                }
                visited.insert(i);
            }
            this_builder.append(true);
        }
        this_builder.finish()
    }

    /// Construct an Arrow schema from this chunk.
    pub fn to_schema(
        &self,
        series_index_name: impl Into<String>,
        encodings: &[ChunkingStrategy],
    ) -> Schema {
        let curie_fields: Vec<FieldRef> =
            serde_arrow::schema::SchemaLike::from_type::<CURIE>(Default::default()).unwrap();

        let base_name = self.chunk_axis.clone();

        let field_meta = base_name.as_field_metadata();
        let mut fields_of = vec![
            Field::new(series_index_name, DataType::UInt64, true).into(),
            Field::new(
                format!("{}_chunk_start", base_name),
                DataType::Float64,
                true,
            )
            .into(),
            Field::new(format!("{}_chunk_end", base_name), DataType::Float64, true).into(),
            Field::new(
                format!("{}_chunk_values", base_name),
                DataType::LargeList(Arc::new(Field::new(
                    "item",
                    array_to_arrow_type(base_name.dtype),
                    true,
                ))),
                true,
            )
            .with_metadata(field_meta)
            .into(),
            Field::new(
                "chunk_encoding",
                DataType::Struct(curie_fields.into()),
                true,
            )
            .into(),
        ];

        for buffer_name in self.arrays.keys().sorted() {
            let f_of = buffer_name.to_field();
            let dtype =
                DataType::LargeList(Arc::new(Field::new("item", f_of.data_type().clone(), true)));
            fields_of.push(Arc::new((*f_of).clone().with_data_type(dtype)));
        }

        for enc in encodings.iter() {
            fields_of.extend(enc.extra_arrays(&base_name).into_iter().map(Arc::new));
        }
        Schema::new(fields_of)
    }

    /// Construct a series of [`ArrowArrayChunk`]s from a [`BinaryArrayMap`], using a specific array indicated by
    /// [`BufferName`] as the main axis, split and encoded using `chunk_encoding`. This may include a set of
    /// transforms according to `drop_zero_intensity`, `nullify_zero_intensity`.
    ///
    /// If `fields` is provided, any array not found in it will be returned as a [`AuxiliaryArray`].
    pub fn from_arrays(
        series_index: u64,
        main_axis: BufferName,
        arrays: &BinaryArrayMap,
        chunk_encoding: ChunkingStrategy,
        overrides: &HashMap<BufferName, BufferName>,
        drop_zero_intensity: bool,
        nullify_zero_intensity: bool,
        fields: Option<&Fields>,
    ) -> Result<(Vec<Self>, Vec<AuxiliaryArray>), ArrayRetrievalError> {
        let mut chunks = Vec::new();
        let mut subset_arrays = BinaryArrayMap::new();

        let mut arrow_arrays = Vec::new();
        let mut intensity_idx = None;
        let mut mz_idx = None;

        let mut auxiliary_arrays = Vec::new();

        for (i, (_, arr)) in arrays.iter().enumerate() {
            let name = BufferName::from_data_array(main_axis.context, arr);
            let buffer_name = if name.array_type == main_axis.array_type {
                &main_axis
            } else {
                overrides.get(&name).unwrap_or(&name)
            };
            if let Some(fields) = fields {
                /// If the buffer isn't in the fields for this chunk schema, skip it and store an auxiliary array.
                if !fields
                    .find(buffer_name.to_field().name()).is_some() && *buffer_name != main_axis {
                        log::debug!("Skipping {:?}, not in schema: {fields:?}", buffer_name.to_field().name());
                        auxiliary_arrays.push(AuxiliaryArray::from_data_array(arr)?);
                        continue
                    }
            }
            if matches!(buffer_name.array_type, ArrayType::IntensityArray) {
                intensity_idx = Some(i);
            } else if matches!(buffer_name.array_type, ArrayType::MZArray) {
                mz_idx = Some(i);
            }
            let array = data_array_to_arrow_array(buffer_name, arr)?;
            arrow_arrays.push((buffer_name.clone(), array));
        }

        if let Some(intensity_idx) = intensity_idx {
            let (intensity_name, intensity_array) = arrow_arrays.get(intensity_idx).unwrap();
            if drop_zero_intensity {
                let (kept_indices, n) = match array_to_arrow_type(intensity_name.dtype) {
                    DataType::Float32 => {
                        let intensity_array = intensity_array.as_primitive::<Float32Type>();
                        (_skip_zero_runs_gen(&intensity_array), intensity_array.len())
                    }
                    DataType::Float64 => {
                        let intensity_array = intensity_array.as_primitive::<Float64Type>();
                        (_skip_zero_runs_gen(&intensity_array), intensity_array.len())
                    }
                    DataType::Int32 => {
                        let intensity_array = intensity_array.as_primitive::<Int32Type>();
                        (_skip_zero_runs_gen(&intensity_array), intensity_array.len())
                    }
                    DataType::Int64 => {
                        let intensity_array = intensity_array.as_primitive::<Int64Type>();
                        (_skip_zero_runs_gen(&intensity_array), intensity_array.len())
                    }
                    _ => {
                        unimplemented!("{}", intensity_name)
                    }
                };
                let kept_indices: UInt64Array = kept_indices.into();
                for (k, v) in arrow_arrays.iter_mut() {
                    if v.len() != n {
                        continue;
                    }
                    *v = arrow::compute::take(v, &kept_indices, None).unwrap();
                }
            }

            if let Some(mz_idx) = mz_idx {
                if nullify_zero_intensity {
                    let (intensity_name, intensity_array) =
                        arrow_arrays.get(intensity_idx).unwrap();
                    let (masked, n) = match array_to_arrow_type(intensity_name.dtype) {
                        DataType::Float32 => {
                            let intensity_array = intensity_array.as_primitive::<Float32Type>();
                            (is_zero_pair_mask(&intensity_array), intensity_array.len())
                        }
                        DataType::Float64 => {
                            let intensity_array = intensity_array.as_primitive::<Float64Type>();
                            (is_zero_pair_mask(&intensity_array), intensity_array.len())
                        }
                        DataType::Int32 => {
                            let intensity_array = intensity_array.as_primitive::<Int32Type>();
                            (is_zero_pair_mask(&intensity_array), intensity_array.len())
                        }
                        DataType::Int64 => {
                            let intensity_array = intensity_array.as_primitive::<Int64Type>();
                            (is_zero_pair_mask(&intensity_array), intensity_array.len())
                        }
                        _ => {
                            unimplemented!("{}", intensity_name)
                        }
                    };

                    let (_, intensities) = arrow_arrays.get_mut(intensity_idx).unwrap();
                    *intensities = nullif::nullif(&intensities.clone(), &masked).unwrap();

                    let (_, mzs) = arrow_arrays.get_mut(mz_idx).unwrap();
                    *mzs = nullif::nullif(&mzs.clone(), &masked).unwrap();
                }
            }
        }

        let main_axis = overrides.get(&main_axis).unwrap_or(&main_axis);

        let (_, main_axis_array) = arrow_arrays
            .iter()
            .find(|(k, v)| k == main_axis)
            .ok_or_else(|| ArrayRetrievalError::NotFound(main_axis.array_type.clone()))?;

        let steps = match array_to_arrow_type(main_axis.dtype) {
            DataType::Float32 => null_chunk_every_k(
                main_axis_array.as_primitive::<Float32Type>(),
                NumCast::from(chunk_encoding.chunk_size()).unwrap(),
            ),
            DataType::Float64 => null_chunk_every_k(
                main_axis_array.as_primitive::<Float64Type>(),
                NumCast::from(chunk_encoding.chunk_size()).unwrap(),
            ),
            _ => unimplemented!("{}", main_axis),
        };

        let mut main_axis = main_axis.clone().with_format(BufferFormat::Chunked);

        for step in steps {
            let slice = main_axis_array.slice(step.start, step.end - step.start);
            let (chunk_start, chunk_end, chunk_values) = match array_to_arrow_type(main_axis.dtype)
            {
                DataType::Float32 => {
                    chunk_encoding.encode_arrow(slice.as_primitive::<Float32Type>())
                }
                DataType::Float64 => {
                    chunk_encoding.encode_arrow(slice.as_primitive::<Float64Type>())
                }
                _ => unimplemented!("{}", main_axis),
            };

            let chunk_arrays: HashMap<BufferName, ArrayRef> = arrow_arrays
                .iter()
                .filter(|(k, _)| k.array_type != main_axis.array_type)
                .map(|(k, v)| {
                    (
                        k.clone().with_format(BufferFormat::ChunkedSecondary),
                        v.slice(step.start, step.end - step.start),
                    )
                })
                .collect();

            chunks.push(Self::new(
                series_index,
                chunk_start,
                chunk_end,
                main_axis.clone(),
                chunk_values,
                chunk_encoding,
                chunk_arrays,
            ));
        }

        Ok((chunks, auxiliary_arrays))
    }
}


#[cfg(test)]
mod test {
    use std::{fs, io};

    use super::*;
    use mzdata::{MZReader, spectrum::MultiLayerSpectrum};

    fn load_chunking_data() -> io::Result<Vec<f64>> {
        let mut reader = io::BufReader::new(fs::File::open("test/data/chunking_mzs.txt")?);

        let mut mzs: Vec<f64> = Vec::new();
        for line in reader.lines().flatten() {
            if line.is_empty() {
                continue;
            }
            mzs.push(line.parse().unwrap());
        }

        Ok(mzs)
    }

    #[test]
    fn test_chunking() -> io::Result<()> {
        let mzs = load_chunking_data()?;

        let intervals = chunk_every_k(&mzs, 10.0);

        let mut last = 0.0;
        for iv in intervals.iter() {
            let vs = &mzs[iv.start..iv.end];
            let term = vs.last().copied().unwrap();
            assert!(
                (term - 1.0) > last,
                "{vs:?} was not more than 9 away from {last}"
            );
            last = term;
        }
        Ok(())
    }

    fn get_arrays_from_mzml() -> io::Result<BinaryArrayMap> {
        let mut reader = MZReader::open_path("small.mzML")?;
        let spec = reader.get_spectrum_by_index(0).unwrap();
        Ok(spec.arrays.clone().unwrap())
    }

    #[test]
    fn test_encode_arrow_drop_zeros() -> io::Result<()> {
        let arrays = get_arrays_from_mzml()?;
        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float32,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &HashMap::new(),
            true,
            false,
            None,
        )?;

        for chunk in chunks.iter() {
            let n = chunk.chunk_values.len();
            for (k, v) in chunk.arrays.iter() {
                assert_eq!(v.len(), n + 1);
            }
        }

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            "spectrum_index",
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
        );

        eprintln!("{:?}", rendered.slice(0, 1));

        Ok(())
    }

    #[test]
    fn test_encode_arrow_drop_zeros_null() -> io::Result<()> {
        let arrays = get_arrays_from_mzml()?;
        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float32,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &HashMap::new(),
            true,
            true,
            None,
        )?;

        for chunk in chunks.iter() {
            let n = chunk.chunk_values.len();
            for (k, v) in chunk.arrays.iter() {
                assert_eq!(v.len(), chunk.arrays.values().next().unwrap().len());
            }
        }

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            "spectrum_index",
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
        );
        Ok(())
    }

    #[test]
    fn test_encode_arrow() -> io::Result<()> {
        let arrays = get_arrays_from_mzml()?;
        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float32,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &HashMap::new(),
            false,
            false,
            None,
        )?;

        let schema = chunks[0].to_schema(
            "spectrum_index",
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
        );

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            "spectrum_index",
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
        );

        eprintln!("{:?}", rendered.slice(0, 1));

        Ok(())
    }
}
