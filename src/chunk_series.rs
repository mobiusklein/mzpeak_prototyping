use std::collections::{HashMap, HashSet};
use std::ops::AddAssign;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, ArrowPrimitiveType, AsArray, Float32Array, Float32Builder,
    Float64Array, Float64Builder, Int32Builder, Int64Builder, LargeListBuilder, PrimitiveArray,
    StructArray, StructBuilder, UInt8Array, UInt8Builder, UInt64Array, UInt64Builder,
};
use arrow::compute::kernels::nullif;
use arrow::datatypes::{
    DataType, Field, Fields, Float32Type, Float64Type, Int32Type, Int64Type, Schema, UInt8Type,
};
use itertools::Itertools;
use mzdata::params::CURIE;
use mzdata::prelude::ByteArrayView;
use mzdata::spectrum::{
    ArrayType, BinaryArrayMap, BinaryDataArrayType, DataArray, bindata::ArrayRetrievalError,
};

use mzpeaks::coordinate::SimpleInterval;

use bytemuck::Pod;
use mzpeaks::prelude::Span1D;
use num_traits::{Float, NumCast, ToPrimitive};

use crate::buffer_descriptors::{BufferOverrideTable, BufferPriority};
use crate::writer::StructVisitor;
use crate::{
    buffer_descriptors::BufferTransform,
    filter::{
        _skip_zero_runs_gen, RegressionDeltaModel, fill_nulls_for, is_zero_pair_mask,
        null_chunk_every_k, null_delta_decode, null_delta_encode,
    },
    peak_series::{
        BufferContext, BufferFormat, BufferName, array_to_arrow_type, data_array_to_arrow_array,
    },
    spectrum::AuxiliaryArray,
    writer::{CURIEBuilder, VisitorBase},
};

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
    accumulator.push(state).unwrap();
    for val in it.iter().copied() {
        state += val;
        accumulator.push(state).unwrap();
    }
}

pub const NO_COMPRESSION: CURIE = mzdata::curie!(MS:1000576);
pub const DELTA_ENCODE: CURIE = mzdata::curie!(MS:1003089);
pub const NUMPRESS_LINEAR: CURIE = mzdata::curie!(MS:1002312);
pub const NUMPRESS_SLOF: CURIE = mzdata::curie!(MS:1002314);

/// Different methods for encoding chunks along a coordinate dimension
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkingStrategy {
    /// Values are encoded as-is without any transformation. While this doesn't have any compression
    /// benefits, it provides compatibility for sparse data. The start and end values are included in
    /// the encoded data.
    Basic { chunk_size: f64 },
    /// Values are stored as deltas w.r.t. previous values. The first value stored is w.r.t. to the chunk
    /// starting value, but the starting value is not encoded in the chunk values array itself. Decoding
    /// of the chunk values requires the chunk start value be read from the chunk metadata.
    Delta { chunk_size: f64 },
    /// Values are stored using MS-Numpress linear compression. The entire chunk is encoded in a byte buffer
    /// which may not align with a multi-byte value type and must be stored in a dedicated byte array. The
    /// start and end values are included in the encoded chunk as well as the chunk metadata.
    NumpressLinear { chunk_size: f64 },
}

impl ChunkingStrategy {
    /// Convert the chunking stategy to a CURIE
    pub const fn as_curie(&self) -> CURIE {
        match self {
            Self::Basic { chunk_size: _ } => NO_COMPRESSION,
            Self::Delta { chunk_size: _ } => DELTA_ENCODE,
            Self::NumpressLinear { chunk_size: _ } => NUMPRESS_LINEAR,
        }
    }

    /// Given the name of the main axis for this chunk, generate all extra [`arrow::datatypes::Field`]
    /// instances needed to populate the schema to include this chunk type.
    pub fn extra_arrays(&self, main_axis_name: &BufferName) -> Vec<Field> {
        match self {
            ChunkingStrategy::Basic { chunk_size: _ } => vec![],
            ChunkingStrategy::Delta { chunk_size: _ } => vec![],
            ChunkingStrategy::NumpressLinear { chunk_size: _ } => {
                let meta = BufferName::new(
                    main_axis_name.context,
                    ArrayType::nonstandard(format!("{}_numpress_bytes", main_axis_name)),
                    main_axis_name.dtype,
                )
                .with_format(BufferFormat::Chunked)
                .with_transform(Some(BufferTransform::NumpressLinear))
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

    /// Encode any extra data arrays from this [`ArrowArrayChunk`] beyond the `chunk_values` array.
    ///
    /// This exists primarily to serve `Numpress` encoding right now, but future encodings might need
    /// it too.
    pub fn encode_extra_arrow(
        &self,
        main_axis_name: &BufferName,
        chunk: &ArrowArrayChunk,
        chunk_builder: &mut StructBuilder,
        schema: &Schema,
        visited: &mut HashSet<usize>,
    ) {
        match self {
            ChunkingStrategy::Basic { chunk_size: _ } => {}
            ChunkingStrategy::Delta { chunk_size: _ } => {}
            ChunkingStrategy::NumpressLinear { chunk_size: _ } => {
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

    /// Get the step size along the main axis for the chunking strategy
    pub const fn chunk_size(&self) -> f64 {
        match self {
            ChunkingStrategy::Basic { chunk_size } => *chunk_size,
            ChunkingStrategy::Delta { chunk_size } => *chunk_size,
            ChunkingStrategy::NumpressLinear { chunk_size } => *chunk_size,
        }
    }

    /// Encode a chunk of an [`PrimitiveArray`] into minimal chunk start, end, and chunk values.
    ///
    /// Assumes the provided `array` has already been cut as a chunk of the desired width.
    ///
    /// # Note
    /// If the array is empty or all null, start and end will be 0.0 and chunk values may be empty.
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
            ChunkingStrategy::Basic { chunk_size: _ } => (start, end, Arc::new(array.clone())),
            ChunkingStrategy::Delta { chunk_size: _ } => {
                (start, end, Arc::new(null_delta_encode(array)))
            }
            ChunkingStrategy::NumpressLinear { chunk_size: _ } => {
                let bytes_of = if matches!(array.data_type(), DataType::Float64) {
                    let array: &PrimitiveArray<Float64Type> =
                        array.as_any().downcast_ref().unwrap();
                    DataArray::compress_numpress_linear(array.values()).unwrap()
                } else {
                    let values: Vec<_> = array
                        .iter()
                        .map(|v| v.and_then(|v| v.to_f64()).unwrap_or_default())
                        .collect();
                    DataArray::compress_numpress_linear(&values).unwrap()
                };
                let array = Arc::new(UInt8Array::from(bytes_of));
                (start, end, array)
            }
        }
    }

    /// Decode the encoded main axis of a chunk into a [`DataArray`].
    ///
    /// Requires the `start_value` of the chunk to provide context.
    ///
    /// ## Warning
    /// If the [`DataArray`] stored type is not bit-level compatible
    /// with the data type that `array` contains or is decoded into,
    /// the data will be meaningless.
    pub fn decode_arrow(
        &self,
        array: &ArrayRef,
        start_value: f64,
        end_value: f64,
        accumulator: &mut DataArray,
        delta_model: Option<&RegressionDeltaModel<f64>>,
    ) {
        if start_value == 0.0 && end_value == 0.0 {
            return;
        }
        macro_rules! decode_delta {
            ($array:ident, $dtype:ty, $native:ty, $debug:literal) => {
                let it = $array.as_primitive::<$dtype>();
                if it.null_count() > 0 {
                    let decoded = null_delta_decode(it, start_value as $native);
                    if let Some(delta_model) = delta_model {
                        let values = fill_nulls_for(&decoded, delta_model);
                        accumulator.extend(&values).unwrap();
                    } else {
                        log::debug!($debug);
                        accumulator.extend(decoded.values()).unwrap()
                    }
                } else {
                    delta_decode(it.values(), start_value as $native, accumulator);
                }
            };
        }

        match self {
            ChunkingStrategy::Basic { chunk_size: _ } => match array.data_type() {
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
            ChunkingStrategy::Delta { chunk_size: _ } => match array.data_type() {
                DataType::Float32 => {
                    decode_delta!(
                        array,
                        Float32Type,
                        f32,
                        "f32 delta decoding contained nulls but no delta model provided"
                    );
                }
                DataType::Float64 => {
                    decode_delta!(
                        array,
                        Float64Type,
                        f64,
                        "f64 delta decoding contained nulls but no delta model provided"
                    );
                }
                _ => panic!(
                    "Data type {:?} is not supported by chunk decoding",
                    array.data_type()
                ),
            },
            ChunkingStrategy::NumpressLinear { chunk_size: _ } => match array.data_type() {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTransformEncoder(BufferTransform);

impl TryFrom<Option<BufferTransform>> for BufferTransformEncoder {
    type Error = <Self as TryFrom<BufferTransform>>::Error;

    fn try_from(value: Option<BufferTransform>) -> Result<Self, Self::Error> {
        match value {
            Some(value) => value.try_into(),
            None => Err(format!("Cannot convert from empty")),
        }
    }
}

impl TryFrom<BufferTransform> for BufferTransformEncoder {
    type Error = String;

    fn try_from(value: BufferTransform) -> Result<Self, Self::Error> {
        match value {
            BufferTransform::NumpressLinear => Ok(Self(value)),
            BufferTransform::NumpressSLOF => Ok(Self(value)),
            BufferTransform::NumpressPIC => Ok(Self(value)),
            BufferTransform::NullInterpolate | BufferTransform::NullZero => Err(format!("{value:?} does not have an encoder"))
        }
    }
}

impl BufferTransformEncoder {
    pub fn to_buffer_name(&self, buffer_name: &BufferName) -> BufferName {
        match self.0 {
            BufferTransform::NumpressLinear => todo!(),
            BufferTransform::NumpressSLOF => BufferName::new(
                buffer_name.context,
                ArrayType::nonstandard(format!("{}_numpress_slof_bytes", buffer_name)),
                buffer_name.dtype,
            )
            .with_format(BufferFormat::ChunkedSecondary)
            .with_transform(Some(self.0)),
            BufferTransform::NumpressPIC => BufferName::new(
                buffer_name.context,
                ArrayType::nonstandard(format!("{}_numpress_pic_bytes", buffer_name)),
                buffer_name.dtype,
            )
            .with_format(BufferFormat::ChunkedSecondary)
            .with_transform(Some(self.0)),
            _ => unimplemented!("{:?} does not have a buffer name", self.0)
        }
    }

    pub fn to_field(&self, buffer_name: &BufferName) -> Field {
        match self.0 {
            BufferTransform::NumpressLinear => todo!(),
            BufferTransform::NumpressSLOF => {
                let meta = buffer_name.as_field_metadata();
                let bytes = Field::new(
                    format!("{}_numpress_slof_bytes", buffer_name),
                    DataType::LargeList(Arc::new(Field::new("item", DataType::UInt8, false))),
                    true,
                )
                .with_metadata(meta);
                bytes
            }
            BufferTransform::NumpressPIC => {
                let meta = buffer_name.as_field_metadata();
                let bytes = Field::new(
                    format!("{}_numpress_pic_bytes", buffer_name),
                    DataType::LargeList(Arc::new(Field::new("item", DataType::UInt8, false))),
                    true,
                )
                .with_metadata(meta);
                bytes
            }
            _ => unimplemented!("{:?} does not have a field conversion", self.0)
        }
    }

    pub fn visit_builder(
        &self,
        buffer_name: &BufferName,
        chunk: &ArrowArrayChunk,
        chunk_builder: &mut StructBuilder,
        schema: &Schema,
        visited: &mut HashSet<usize>,
    ) {
        // The buffer name's array is already the name of the field we want, we don't want to
        // use the full buffer name formatting again
        let f = self.to_field(buffer_name);
        let q = f.name();
        let idx = schema.fields().iter().position(|p| p.name() == q).unwrap();

        if visited.contains(&idx) {
            return;
        }
        visited.insert(idx);
        let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
            chunk_builder.field_builder(idx).unwrap();

        if let Some(chunk_segment) = chunk.arrays.get(buffer_name) {
            let inner = b
                .values()
                .as_any_mut()
                .downcast_mut::<UInt8Builder>()
                .unwrap();
            let bytes: &UInt8Array = chunk_segment.as_primitive();
            inner.extend(bytes);
            b.append(true);
        } else {
            b.append_null();
        }
    }

    pub fn encode_arrow(
        &self,
        _buffer_name: &BufferName,
        chunk_segment: &impl AsArray,
    ) -> ArrayRef {
        match self.0 {
            BufferTransform::NumpressLinear => todo!(),
            BufferTransform::NumpressPIC => {
                let mut bytes = Vec::new();
                if let Some(vals) = chunk_segment.as_primitive_opt::<Float32Type>() {
                    let vals = vals.values();
                    numpress_rs::encode_pic(&vals, &mut bytes).unwrap();
                } else if let Some(vals) = chunk_segment.as_primitive_opt::<Float64Type>() {
                    let vals = vals.values();
                    numpress_rs::encode_pic(&vals, &mut bytes).unwrap();
                } else {
                    todo!()
                }
                Arc::new(UInt8Array::from(bytes))
            }
            BufferTransform::NumpressSLOF => {
                let mut bytes = Vec::new();
                if let Some(vals) = chunk_segment.as_primitive_opt::<Float32Type>() {
                    let vals = vals.values();
                    let fp = numpress_rs::optimal_slof_fixed_point(&vals);
                    numpress_rs::encode_slof(&vals, &mut bytes, fp).unwrap();
                } else if let Some(vals) = chunk_segment.as_primitive_opt::<Float64Type>() {
                    let vals = vals.values();
                    let fp = numpress_rs::optimal_slof_fixed_point(&vals);
                    numpress_rs::encode_slof(&vals, &mut bytes, fp).unwrap();
                } else {
                    todo!()
                }
                let bytes = UInt8Array::from(bytes);
                Arc::new(bytes)
            }
            _ => unimplemented!("{:?} does not have an encoder", self.0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTransformDecoder(BufferTransform);

impl BufferTransformDecoder {
    pub fn decode(&self, buffer_name: &BufferName, array: &impl AsArray) -> ArrayRef {
        match self.0 {
            BufferTransform::NumpressLinear => todo!(),
            BufferTransform::NumpressSLOF => {
                let data: &UInt8Array = array.as_primitive();
                match buffer_name.dtype {
                    BinaryDataArrayType::Float64 => {
                        let mut acc: Vec<f64> = Vec::new();
                        numpress_rs::decode_slof(data.values(), &mut acc).unwrap();
                        return Arc::new(Float64Array::from(acc));
                    }
                    BinaryDataArrayType::Float32 => {
                        let mut acc: Vec<f64> = Vec::new();
                        numpress_rs::decode_slof(data.values(), &mut acc).unwrap();
                        return Arc::new(Float32Array::from_iter_values(
                            acc.into_iter().map(|v| v as f32),
                        ));
                    }
                    _ => todo!(),
                }
            }
            BufferTransform::NumpressPIC => {
                let data: &UInt8Array = array.as_primitive();
                match buffer_name.dtype {
                    BinaryDataArrayType::Float64 => {
                        let mut acc: Vec<f64> = Vec::new();
                        numpress_rs::decode_pic(data.values(), &mut acc).unwrap();
                        return Arc::new(Float64Array::from(acc));
                    }
                    BinaryDataArrayType::Float32 => {
                        let mut acc: Vec<f64> = Vec::new();
                        numpress_rs::decode_pic(data.values(), &mut acc).unwrap();
                        return Arc::new(Float32Array::from_iter_values(
                            acc.into_iter().map(|v| v as f32),
                        ));
                    }
                    _ => todo!(),
                }
            }
            _ => unimplemented!("{:?} does not have a decoder", self.0)
        }
    }
}

impl TryFrom<Option<BufferTransform>> for BufferTransformDecoder {
    type Error = <Self as TryFrom<BufferTransform>>::Error;

    fn try_from(value: Option<BufferTransform>) -> Result<Self, Self::Error> {
        match value {
            Some(value) => value.try_into(),
            None => Err(format!("Cannot convert from empty")),
        }
    }
}

impl TryFrom<BufferTransform> for BufferTransformDecoder {
    type Error = String;

    fn try_from(value: BufferTransform) -> Result<Self, Self::Error> {
        match value {
            BufferTransform::NumpressLinear => Ok(Self(value)),
            BufferTransform::NumpressSLOF => Ok(Self(value)),
            BufferTransform::NumpressPIC => Ok(Self(value)),
            BufferTransform::NullInterpolate | BufferTransform::NullZero => Err(format!("{value:?} does not have a decoder"))
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrowArrayChunk {
    /// The index of source entity
    pub series_index: u64,
    series_time: Option<f32>,
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

impl Span1D for ArrowArrayChunk {
    type DimType = f64;

    fn start(&self) -> Self::DimType {
        self.chunk_start
    }

    fn end(&self) -> Self::DimType {
        self.chunk_end
    }
}

impl ArrowArrayChunk {
    /// Low level constructor for a single chunk record.
    ///
    /// Prefer [`ArrowArrayChunk::from_arrays`] for constructing a block of [`ArrowArrayChunk`]
    pub fn new(
        series_index: u64,
        series_time: Option<f32>,
        chunk_start: f64,
        chunk_end: f64,
        chunk_axis: BufferName,
        chunk_values: ArrayRef,
        chunk_encoding: ChunkingStrategy,
        arrays: HashMap<BufferName, ArrayRef>,
    ) -> Self {
        Self {
            series_index,
            series_time,
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
        buffer_context: BufferContext,
        _schema: &Fields,
        encodings: &[ChunkingStrategy],
        include_time: bool,
    ) -> StructArray {
        let this_schema = chunks[0].to_schema(buffer_context, encodings, include_time);
        let mut this_builder =
            StructBuilder::from_fields(this_schema.fields().clone(), chunks.len());

        this_builder.field_builders_mut()[4] =
            Box::new(CURIEBuilder::default()) as Box<dyn ArrayBuilder>;

        let time_index = if include_time {
            let q = buffer_context.time_field();
            let q = q.name();
            this_schema
                .fields()
                .iter()
                .position(|f| *f.name() == *q)
                .unwrap()
        } else {
            0
        };

        let mut visited: HashSet<usize> = HashSet::new();
        for chunk in chunks {
            let mut field_i = 0;
            visited.clear();
            let b = this_builder
                .field_builder::<UInt64Builder>(field_i)
                .unwrap();
            b.append_value(chunk.series_index);
            visited.insert(field_i);
            field_i += 1;
            let b = this_builder
                .field_builder::<Float64Builder>(field_i)
                .unwrap();
            b.append_value(chunk.chunk_start);
            visited.insert(field_i);
            field_i += 1;
            let b = this_builder
                .field_builder::<Float64Builder>(field_i)
                .unwrap();
            b.append_value(chunk.chunk_end);
            visited.insert(field_i);
            field_i += 1;

            let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                this_builder.field_builder(field_i).unwrap();
            if matches!(
                chunk.chunk_encoding,
                ChunkingStrategy::NumpressLinear { chunk_size: _ }
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
            visited.insert(field_i);
            field_i += 1;
            for encoding in encodings {
                encoding.encode_extra_arrow(
                    &chunk.chunk_axis,
                    &chunk,
                    &mut this_builder,
                    &this_schema,
                    &mut visited,
                );
            }

            let cb = this_builder.field_builder::<CURIEBuilder>(field_i).unwrap();
            let curie_of = chunk.chunk_encoding.as_curie();
            cb.append_value(&curie_of);
            visited.insert(field_i);
            field_i += 1;

            if include_time {
                let b = this_builder
                    .field_builder::<Float32Builder>(time_index)
                    .unwrap();
                b.append_option(chunk.series_time);
                visited.insert(time_index);
            }

            for (i, f) in this_schema.fields().iter().enumerate().skip(field_i) {
                if visited.contains(&i) {
                    continue;
                }
                if let Some(buf_name) = BufferName::from_field(chunk.chunk_axis.context, f.clone())
                    .map(|f| f.with_format(BufferFormat::ChunkedSecondary))
                {
                    let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                        this_builder.field_builder(i).unwrap();

                    if let Some(transform) = BufferTransformEncoder::try_from(buf_name.transform).ok()
                    {
                        transform.visit_builder(
                            &buf_name,
                            chunk,
                            &mut this_builder,
                            &this_schema,
                            &mut visited,
                        );
                    } else {
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
    ///
    /// This schema must hold for just *this* block of chunks. It will
    /// be adapted by [`ChunkBuffers`](crate::writer::ChunkBuffers) to
    /// the file-level schema.
    pub fn to_schema(
        &self,
        buffer_context: BufferContext,
        encodings: &[ChunkingStrategy],
        include_time: bool,
    ) -> Schema {
        let base_name = self.chunk_axis.clone();
        let mut bounds_name = base_name.clone();
        bounds_name.dtype = BinaryDataArrayType::Float64;
        let (start, end) = bounds_name.make_bounds_fields().unwrap();
        let field_meta = base_name.as_field_metadata();
        let chunk_encoding_meta = base_name
            .clone()
            .with_format(BufferFormat::ChunkEncoding)
            .as_field_metadata();
        let mut fields_of = vec![
            buffer_context.index_field(),
            start,
            end,
            Field::new(
                base_name.to_string(),
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
                CURIEBuilder::default().as_struct_type(),
                true,
            )
            .with_metadata(chunk_encoding_meta)
            .into(),
        ];

        for buffer_name in self.arrays.keys().sorted() {
            if let Ok(transform) = BufferTransformEncoder::try_from(buffer_name.transform) {
                let f_of = transform.to_field(buffer_name);
                fields_of.push(Arc::new(f_of));
            } else {
                let f_of = buffer_name.to_field();
                let dtype = DataType::LargeList(Arc::new(Field::new(
                    "item",
                    f_of.data_type().clone(),
                    true,
                )));
                fields_of.push(Arc::new((*f_of).clone().with_data_type(dtype)));
            }
        }

        for enc in encodings.iter() {
            fields_of.extend(enc.extra_arrays(&base_name).into_iter().map(Arc::new));
        }
        if include_time {
            fields_of.push(buffer_context.time_field());
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
        series_time: Option<f32>,
        main_axis: BufferName,
        arrays: &BinaryArrayMap,
        chunk_encoding: ChunkingStrategy,
        overrides: &BufferOverrideTable,
        drop_zero_intensity: bool,
        nullify_zero_intensity: bool,
        fields: Option<&Fields>,
    ) -> Result<(Vec<Self>, Vec<AuxiliaryArray>), ArrayRetrievalError> {
        let mut chunks = Vec::new();

        let mut arrow_arrays = Vec::new();
        let mut intensity_idx = None;
        let mut mz_idx = None;

        let mut auxiliary_arrays = Vec::new();

        let main_axis = overrides
            .map(&main_axis)
            .with_format(BufferFormat::Chunked)
            .with_priority(Some(BufferPriority::Primary))
            .with_sorting_rank(Some(0));

        // Ensure that non-hashing properties of [`BufferName`] propagate from the
        // schema down to physical arrays constructed. Also propagate any transformations
        // in the schema, which *are* hash-dependent but considered safe here.
        let mut fields_of = BufferOverrideTable::default();
        if let Some(fields) = fields.as_ref() {
            for f in fields.iter() {
                if let Some(f) = BufferName::from_field(main_axis.context, f.clone()) {
                    fields_of.insert(f.clone(), f.clone());
                    if f.transform.is_some() {
                        fields_of.insert(f.clone().with_transform(None), f.clone());
                    }
                }
            }
        }

        let empty_main_axis = match arrays.get(&main_axis.array_type) {
            Some(v) => v.raw_len() == 0,
            None => true,
        };

        if empty_main_axis {
            for arr in arrays
                .iter()
                .filter_map(|(_, arr)| (arr.raw_len() > 0).then(|| arr))
            {
                auxiliary_arrays.push(AuxiliaryArray::from_data_array(arr)?);
            }
            return Ok((Vec::new(), auxiliary_arrays));
        }

        for (i, (_, arr)) in arrays.iter().enumerate() {
            let name = BufferName::from_data_array(main_axis.context, arr);
            let buffer_name0 = if name.array_type == main_axis.array_type {
                main_axis.clone().with_format(BufferFormat::Chunked)
            } else {
                overrides
                    .map(&name)
                    .with_format(BufferFormat::ChunkedSecondary)
            };
            let buffer_name = fields_of.map(&buffer_name0);

            if let Some(fields) = fields {
                let field_name = if let Ok(transform) =
                    BufferTransformEncoder::try_from(buffer_name.transform)
                {
                    transform.to_field(&buffer_name).name().clone()
                } else {
                    buffer_name.to_field().name().clone()
                };
                // If the buffer isn't in the fields for this chunk schema, skip it and store an auxiliary array.
                if !fields.find(&field_name).is_some() && buffer_name != main_axis {
                    log::debug!(
                        "Skipping {:?} from {arr:?}, not in schema: {fields:?}",
                        buffer_name.to_field().name()
                    );
                    auxiliary_arrays.push(AuxiliaryArray::from_data_array(arr)?);
                    continue;
                }
            }
            if matches!(buffer_name.array_type, ArrayType::IntensityArray) {
                intensity_idx = Some(i);
            } else if matches!(buffer_name.array_type, ArrayType::MZArray) {
                mz_idx = Some(i);
            }
            let array = data_array_to_arrow_array(&buffer_name, arr)?;
            arrow_arrays.push((buffer_name, array));
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
                for (_, v) in arrow_arrays.iter_mut() {
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
                    let (masked, _) = match array_to_arrow_type(intensity_name.dtype) {
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

        let (_, main_axis_array) = match arrow_arrays.iter().find(|(k, _)| *k == main_axis) {
            Some(x) => x,
            None => {
                log::warn!(
                    "Primary axis array is missing ({main_axis}) for {series_index} post-conversion"
                );
                return Ok((Vec::new(), Vec::new()));
            }
        };

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

        let main_axis = main_axis.clone().with_format(BufferFormat::Chunked);

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

            let mut chunk_arrays: HashMap<BufferName, ArrayRef> = Default::default();
            for (k, v) in arrow_arrays
                .iter()
                .filter(|(k, _)| k.array_type != main_axis.array_type)
            {
                let k = k.clone().with_format(BufferFormat::ChunkedSecondary);
                let v = v.slice(step.start, step.end - step.start);
                if let Ok(transform) = BufferTransformEncoder::try_from(k.transform) {
                    let vi = transform.encode_arrow(&k, &v);
                    chunk_arrays.insert(k, vi);
                } else {
                    chunk_arrays.insert(k, v);
                }
            }

            chunks.push(Self::new(
                series_index,
                series_time,
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
    use std::{
        fs,
        io::{self, prelude::*},
    };

    use crate::filter::{MZDeltaModel, drop_where_column_is_zero, nullify_at_zero};

    use super::*;
    use arrow::array::RecordBatch;
    use mzdata::{MZReader, params::Unit, prelude::*};

    fn load_chunking_data() -> io::Result<Vec<f64>> {
        let reader = io::BufReader::new(fs::File::open("test/data/chunking_mzs.txt")?);

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
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &BufferOverrideTable::default(),
            true,
            false,
            None,
        )?;

        for chunk in chunks.iter() {
            let n = chunk.chunk_values.len();
            for (_, v) in chunk.arrays.iter() {
                assert_eq!(v.len(), n + 1);
            }
        }

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            false,
        );

        let f = rendered.column_by_name("mz_chunk_start").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Float64);
        let f = rendered.column_by_name("mz_chunk_end").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Float64);

        let f = rendered.column_by_name("mz_chunk_values").unwrap();
        assert_eq!(f.data_type().clone(), DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))));
        assert_eq!(f.len(), 36);
        let k = f.as_list::<i64>().iter().map(|a| a.unwrap().len()).sum::<usize>();
        assert_eq!(k, 13553);

        let f = rendered.column_by_name("chunk_encoding").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Utf8);

        let f = rendered.column_by_name("intensity_f32_dc").unwrap();
        assert_eq!(f.data_type().clone(), DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))));
        assert_eq!(f.len(), 36);
        let k = f.as_list::<i64>().iter().map(|a| a.unwrap().len()).sum::<usize>();
        assert_eq!(k, 13589);

        Ok(())
    }

    #[test]
    fn test_encode_arrow_drop_zeros_null() -> io::Result<()> {
        let mut arrays = get_arrays_from_mzml()?;
        let arr = arrays.get_mut(&ArrayType::MZArray).unwrap();
        arr.store_as(BinaryDataArrayType::Float64)?;

        let mzs = arrays.mzs()?;
        let intens: Vec<f64> = arrays
            .intensities()?
            .iter()
            .map(|w| (*w).sqrt() as f64)
            .collect();

        let betas = crate::filter::select_delta_model(&mzs, Some(&intens));
        let delta_model = RegressionDeltaModel::<f64>::from_f64_iter(betas.into_iter());

        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float64,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &BufferOverrideTable::default(),
            true,
            true,
            None,
        )?;

        for chunk in chunks.iter() {
            for (_, v) in chunk.arrays.iter() {
                assert_eq!(v.len(), chunk.arrays.values().next().unwrap().len());
            }
        }

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            false,
        );

        let start_values = rendered.column(1).as_primitive::<Float64Type>();
        let end_values = rendered.column(2).as_primitive::<Float64Type>();
        let chunk_values = rendered.column(3).as_list::<i64>();
        let intensity_values = rendered.column(5).as_list::<i64>();

        let intensity_values: Vec<f32> = intensity_values
            .iter()
            .flatten()
            .map(|a| {
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let mut accumulator =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        for ((start_value, end_value), chunk_vals) in start_values
            .iter()
            .flatten()
            .zip(end_values.iter().flatten())
            .zip(chunk_values.iter().flatten())
        {
            ChunkingStrategy::Delta { chunk_size: 50.0 }.decode_arrow(
                &chunk_vals,
                start_value as f64,
                end_value as f64,
                &mut accumulator,
                Some(&delta_model),
            );
        }

        assert_eq!(accumulator.data_len()?, intensity_values.len());

        let mz_array = Float64Array::from_iter_values(mzs.iter().copied());
        let intensity_array = Float32Array::from_iter_values(intens.iter().map(|v| *v as f32));
        let mini_schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("mz_array", DataType::Float64, true)),
            Arc::new(Field::new("intensity_array", DataType::Float32, true)),
        ]));

        let batch = RecordBatch::try_new(
            mini_schema.clone(),
            vec![Arc::new(mz_array) as ArrayRef, Arc::new(intensity_array)],
        )
        .unwrap();

        let trimmed_batch1 = drop_where_column_is_zero(&batch, 1).unwrap();
        let trimmed_batch = nullify_at_zero(&trimmed_batch1, 1, &[0, 1]).unwrap();

        assert_eq!(trimmed_batch.num_rows(), accumulator.data_len()?);

        let mz_acc = accumulator.to_f64()?;
        let mz_ref = trimmed_batch.column(0).as_primitive::<Float64Type>();
        let mz_ref = crate::filter::fill_nulls_for(mz_ref, &delta_model);

        for ((i, a), b) in mz_acc
            .iter()
            .copied()
            .enumerate()
            .zip(mz_ref.iter().copied())
        {
            if intensity_values[i] == 0.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            } else {
                assert_eq!(
                    a,
                    b,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_encode_arrow_transform() -> io::Result<()> {
        let arrays = get_arrays_from_mzml()?;
        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float32,
        );

        let intensity_name = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
        )
        .with_unit(Unit::DetectorCounts);
        let intensity_name_tfm = intensity_name
            .clone()
            .with_transform(Some(BufferTransform::NumpressSLOF));
        let overrides = BufferOverrideTable::from_iter(vec![(intensity_name, intensity_name_tfm)]);

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &overrides,
            false,
            false,
            None,
        )?;

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            false,
        );

        // for col in rendered.column_names() {
        //     eprintln!("{col}, {}", rendered.column_by_name(col).unwrap().len());
        // }

        assert!(
            rendered
                .column_by_name("intensity_f32_dc_numpress_slof_bytes")
                .is_some()
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
        )
        .with_unit(Unit::MZ);

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &Default::default(),
            false,
            false,
            None,
        )?;

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            true,
        );

        let names = rendered.column_names();
        assert!(names.contains(&"spectrum_time"));

        let f = rendered.column_by_name("mz_chunk_start").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Float64);
        let f = rendered.column_by_name("mz_chunk_end").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Float64);
        let f = rendered.column_by_name("mz_chunk_values").unwrap();
        assert_eq!(f.data_type().clone(), DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))));

        assert_eq!(f.len(), 36);
        let k = f.as_list::<i64>().iter().map(|a| a.unwrap().len()).sum::<usize>();
        assert_eq!(k, 19877);

        let f = rendered.column_by_name("chunk_encoding").unwrap();
        assert_eq!(f.data_type().clone(), DataType::Utf8);

        let f = rendered.column_by_name("intensity_f32_dc").unwrap();
        assert_eq!(f.data_type().clone(), DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))));
        assert_eq!(f.len(), 36);
        let k = f.as_list::<i64>().iter().map(|a| a.unwrap().len()).sum::<usize>();
        assert_eq!(k, 19913);


        Ok(())
    }

    #[test]
    fn test_encode_arrow_drop_zeros_null2() -> io::Result<()> {
        let reader = io::BufReader::new(std::fs::File::open("test/data/sparse_large_gaps.txt")?);
        let mut mzs =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        let mut intensities =
            DataArray::from_name_and_type(&ArrayType::IntensityArray, BinaryDataArrayType::Float32);
        for line in reader.lines().flatten() {
            if let Some((a, b)) = line.split_once("\t") {
                mzs.push(a.parse::<f64>().unwrap())?;
                intensities.push(b.parse::<f32>().unwrap())?;
            }
        }

        let mut arrays = BinaryArrayMap::new();
        arrays.add(mzs);
        arrays.add(intensities);

        let mzs = arrays.mzs()?;
        let weights: Vec<f64> = arrays
            .intensities()?
            .iter()
            .map(|w| (*w).sqrt() as f64)
            .collect();

        let betas = crate::filter::select_delta_model(&mzs, Some(&weights));
        let delta_model = RegressionDeltaModel::<f64>::from_f64_iter(betas.into_iter());

        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float64,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &Default::default(),
            true,
            true,
            None,
        )?;

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            false,
        );

        let start_values = rendered.column(1).as_primitive::<Float64Type>();
        let end_values = rendered.column(2).as_primitive::<Float64Type>();
        let chunk_values = rendered.column(3).as_list::<i64>();
        let intensity_values = rendered.column(5).as_list::<i64>();

        let intensity_values: Vec<f32> = intensity_values
            .iter()
            .flatten()
            .map(|a| {
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let mut accumulator =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        for ((start_value, end_value), chunk_vals) in start_values
            .iter()
            .flatten()
            .zip(end_values.iter().flatten())
            .zip(chunk_values.iter().flatten())
        {
            ChunkingStrategy::Delta { chunk_size: 50.0 }.decode_arrow(
                &chunk_vals,
                start_value as f64,
                end_value as f64,
                &mut accumulator,
                Some(&delta_model),
            );
        }

        assert_eq!(accumulator.data_len()?, intensity_values.len());

        let intens = arrays.intensities()?;

        let mz_array = Float64Array::from_iter_values(mzs.iter().copied());
        let intensity_array = Float32Array::from_iter_values(intens.iter().map(|v| *v as f32));
        let mini_schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("mz_array", DataType::Float64, true)),
            Arc::new(Field::new("intensity_array", DataType::Float32, true)),
        ]));

        let batch = RecordBatch::try_new(
            mini_schema.clone(),
            vec![Arc::new(mz_array) as ArrayRef, Arc::new(intensity_array)],
        )
        .unwrap();

        let trimmed_batch1 = drop_where_column_is_zero(&batch, 1).unwrap();
        let trimmed_batch = nullify_at_zero(&trimmed_batch1, 1, &[0, 1]).unwrap();

        assert_eq!(trimmed_batch.num_rows(), accumulator.data_len()?);

        let mz_acc = accumulator.to_f64()?;
        let mz_ref = trimmed_batch.column(0).as_primitive::<Float64Type>();
        let mz_ref = crate::filter::fill_nulls_for(mz_ref, &delta_model);

        for ((i, a), b) in mz_acc
            .iter()
            .copied()
            .enumerate()
            .zip(mz_ref.iter().copied())
        {
            if intensity_values[i] == 0.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            } else {
                assert_eq!(
                    a,
                    b,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_encode_arrow_drop_zeros_null3() -> io::Result<()> {
        let reader = io::BufReader::new(std::fs::File::open("test/data/sparse_sciex.txt")?);
        let mut mzs =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        let mut intensities =
            DataArray::from_name_and_type(&ArrayType::IntensityArray, BinaryDataArrayType::Float32);
        for line in reader.lines().flatten() {
            if let Some((a, b)) = line.split_once("\t") {
                mzs.push(a.parse::<f64>().unwrap())?;
                intensities.push(b.parse::<f32>().unwrap())?;
            }
        }

        let mut arrays = BinaryArrayMap::new();
        arrays.add(mzs);
        arrays.add(intensities);

        let mzs = arrays.mzs()?;
        let weights: Vec<f64> = arrays
            .intensities()?
            .iter()
            .map(|w| (*w).sqrt() as f64)
            .collect();

        let betas = crate::filter::select_delta_model(&mzs, Some(&weights));
        let delta_model = RegressionDeltaModel::<f64>::from_f64_iter(betas.into_iter());

        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float64,
        );

        let (chunks, _) = ArrowArrayChunk::from_arrays(
            0,
            None,
            target,
            &arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            &Default::default(),
            true,
            true,
            None,
        )?;

        let rendered = ArrowArrayChunk::to_struct_array(
            &chunks,
            BufferContext::Spectrum,
            Schema::empty().fields(),
            &[
                ChunkingStrategy::Basic { chunk_size: 50.0 },
                ChunkingStrategy::Delta { chunk_size: 50.0 },
            ],
            false,
        );

        let start_values = rendered.column(1).as_primitive::<Float64Type>();
        let end_values = rendered.column(2).as_primitive::<Float64Type>();
        let chunk_values = rendered.column(3).as_list::<i64>();
        let intensity_values = rendered.column(5).as_list::<i64>();

        let intensity_values: Vec<f32> = intensity_values
            .iter()
            .flatten()
            .map(|a| {
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        let mut accumulator =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        for ((start_value, end_value), chunk_vals) in start_values
            .iter()
            .flatten()
            .zip(end_values.iter().flatten())
            .zip(chunk_values.iter().flatten())
        {
            ChunkingStrategy::Delta { chunk_size: 50.0 }.decode_arrow(
                &chunk_vals,
                start_value as f64,
                end_value as f64,
                &mut accumulator,
                Some(&delta_model),
            );
        }

        assert_eq!(accumulator.data_len()?, intensity_values.len());

        let intens = arrays.intensities()?;

        let mz_array = Float64Array::from_iter_values(mzs.iter().copied());
        let intensity_array = Float32Array::from_iter_values(intens.iter().map(|v| *v as f32));
        let mini_schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("mz_array", DataType::Float64, true)),
            Arc::new(Field::new("intensity_array", DataType::Float32, true)),
        ]));

        let batch = RecordBatch::try_new(
            mini_schema.clone(),
            vec![Arc::new(mz_array) as ArrayRef, Arc::new(intensity_array)],
        )
        .unwrap();

        let trimmed_zero_dropped_batch = drop_where_column_is_zero(&batch, 1).unwrap();
        let trimmed_null_filled_batch =
            nullify_at_zero(&trimmed_zero_dropped_batch, 1, &[0, 1]).unwrap();

        assert_eq!(
            trimmed_null_filled_batch.num_rows(),
            accumulator.data_len()?
        );

        let mz_acc = accumulator.to_f64()?;
        let mz_ref = trimmed_null_filled_batch
            .column(0)
            .as_primitive::<Float64Type>();
        let mz_ref = crate::filter::fill_nulls_for(mz_ref, &delta_model);

        for ((i, a), b) in mz_acc
            .iter()
            .copied()
            .enumerate()
            .zip(mz_ref.iter().copied())
        {
            if intensity_values[i] == 0.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            } else {
                assert_eq!(
                    a,
                    b,
                    "{i}: {} {a} - {b} = {}",
                    intensity_values.get(i).unwrap(),
                    a - b
                );
            }
        }

        Ok(())
    }
}
