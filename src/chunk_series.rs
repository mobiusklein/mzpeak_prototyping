#![allow(unused)]
use std::collections::HashMap;
use std::ops::AddAssign;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, AsArray, Float32Array, Float32Builder, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array, Int64Builder, LargeBinaryArray, LargeListBuilder, StructArray, StructBuilder, UInt32Builder, UInt64Array, UInt64Builder, UInt8Array, UInt8Builder
};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Float32Type, Float64Type, Schema};
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
use num_traits::{Float, NumCast};

use crate::peak_series::{array_to_arrow_type, BufferContext, BufferFormat, BufferName, MZ_ARRAY};
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
            t = t + k;
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

pub fn delta_decode<T: Float + Pod + AddAssign>(it: &[T], start_value: T, accumulator: &mut DataArray) {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkingStrategy {
    Basic { chunk_size: f64, },
    Delta { chunk_size: f64, },
    NumpressLinear { chunk_size: f64, },
}

impl ChunkingStrategy {
    pub fn as_curie(&self) -> CURIE {
        match self {
            Self::Basic { chunk_size } => NO_COMPRESSION,
            Self::Delta { chunk_size } => DELTA_ENCODE,
            Self::NumpressLinear { chunk_size } => NUMPRESS_LINEAR,
        }
    }

    pub fn chunk_size(&self) -> f64 {
        match self {
            ChunkingStrategy::Basic { chunk_size } => *chunk_size,
            ChunkingStrategy::Delta { chunk_size } => *chunk_size,
            ChunkingStrategy::NumpressLinear { chunk_size } => *chunk_size,
        }
    }

    pub fn encode<T: Float + Pod>(&self, data: &[T], name: &ArrayType, dtype: &BinaryDataArrayType) -> (f64, f64, DataArray) {
        match self {
            ChunkingStrategy::Basic { chunk_size } => {
                let mut acc = DataArray::from_name_type_size(name, *dtype, dtype.size_of() * data.len());
                acc.extend(data).unwrap();
                let start = data.first().copied().unwrap_or_else(|| T::zero());
                let end = data.last().copied().unwrap_or_else(|| T::zero());
                (start.to_f64().unwrap(), end.to_f64().unwrap(), acc)
            },
            ChunkingStrategy::Delta { chunk_size } => {
                delta_encode(data, name, dtype)
            },
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                let mut acc = DataArray::from_name_type_size(name, *dtype, dtype.size_of() * data.len());
                acc.extend(data).unwrap();
                acc.store_compressed(BinaryCompressionType::NumpressLinear);
                let start = data.first().copied().unwrap_or_else(|| T::zero());
                let end = data.last().copied().unwrap_or_else(|| T::zero());
                (start.to_f64().unwrap(), end.to_f64().unwrap(), acc)
            },
        }
    }

    pub fn decode_arrow(&self, array: &ArrayRef, start_value: f64, accumulator: &mut DataArray) {
        match self {
            ChunkingStrategy::Basic { chunk_size } => {
                match array.data_type() {
                    DataType::Float32 => {
                        let it = array.as_primitive::<Float32Type>();
                        accumulator.extend(it.values()).unwrap();
                    }
                    DataType::Float64 => {
                        let it = array.as_primitive::<Float64Type>();
                        accumulator.extend(it.values()).unwrap();
                    }
                    _ => panic!("Data type {:?} is not supported by basic decoding", array.data_type())
                }
            },
            ChunkingStrategy::Delta { chunk_size } => {
                match array.data_type() {
                    DataType::Float32 => {
                        let it = array.as_primitive::<Float32Type>();
                        delta_decode(it.values(), start_value as f32, accumulator);
                    }
                    DataType::Float64 => {
                        let it = array.as_primitive::<Float64Type>();
                        delta_decode(it.values(), start_value, accumulator);
                    }
                    _ => panic!("Data type {:?} is not supported by chunk decoding", array.data_type())
                }
            },
            ChunkingStrategy::NumpressLinear { chunk_size } => {
                match array.data_type() {
                    DataType::Float64 => {
                        let it = array.as_primitive::<Float64Type>();
                        todo!("Still working on numpress decoding")
                    }
                    _ => panic!("Data type {:?} is not supported by numpress linear decoding", array.data_type())
                }
            },
        }
    }

    fn decode<T: Float + Pod + AddAssign>(&self, it: &[T], start_value: T, accumulator: &mut DataArray) {

    }
}


#[derive(Debug, Clone)]
pub struct ArrayChunk {
    pub series_index: u64,
    pub chunk_start: f64,
    pub chunk_end: f64,
    pub chunk_values: DataArray,
    pub chunk_encoding: ChunkingStrategy,
    pub arrays: BinaryArrayMap,
}

impl ArrayChunk {
    pub fn new(
        series_index: u64,
        chunk_start: f64,
        chunk_end: f64,
        chunk_values: DataArray,
        chunk_encoding: ChunkingStrategy,
        arrays: BinaryArrayMap,
    ) -> Self {
        Self {
            series_index,
            chunk_start,
            chunk_end,
            chunk_values,
            chunk_encoding,
            arrays,
        }
    }

    pub fn to_arrays(chunks: &[Self],
            series_index_name: impl Into<String>,
            buffer_context: BufferContext,
            schema: &Schema,
            overrides: &HashMap<BufferName, BufferName>
        ) -> Result<StructArray, ArrayRetrievalError> {

        let this_schema = chunks[0].to_schema(series_index_name, buffer_context, overrides);
        let main_axis = BufferName::new(buffer_context, chunks[0].chunk_values.name().clone(), chunks[0].chunk_values.dtype());
        let main_axis = overrides.get(&main_axis).unwrap_or(&main_axis);
        let mut this_builder = StructBuilder::from_fields(this_schema.fields().clone(), chunks.len());
        for chunk in chunks {
            let mut b = this_builder.field_builder::<UInt64Builder>(0).unwrap();
            b.append_value(chunk.series_index);
            let mut b = this_builder.field_builder::<Float64Builder>(1).unwrap();
            b.append_value(chunk.chunk_start);
            let mut b = this_builder.field_builder::<Float64Builder>(2).unwrap();
            b.append_value(chunk.chunk_end);

            let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> = this_builder.field_builder(3).unwrap();
            match array_to_arrow_type(main_axis.dtype) {
                DataType::Int32 => {
                    let inner = b.values().as_any_mut().downcast_mut::<Int32Builder>().unwrap();
                    inner.extend(chunk.chunk_values.to_i32()?.iter().copied().map(Some));
                },
                DataType::Int64 => {
                    let inner = b.values().as_any_mut().downcast_mut::<Int64Builder>().unwrap();
                    inner.extend(chunk.chunk_values.to_i64()?.iter().copied().map(Some));
                },
                DataType::Float32 => {
                    let inner = b.values().as_any_mut().downcast_mut::<Float32Builder>().unwrap();
                    inner.extend(chunk.chunk_values.to_f32()?.iter().copied().map(Some));
                },
                DataType::Float64 => {
                    let inner = b.values().as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                    inner.extend(chunk.chunk_values.to_f64()?.iter().copied().map(Some));
                },
                DataType::LargeBinary => todo!(),
                tp => {
                    unimplemented!("Array type {tp:?} from {:?} not supported", chunk.chunk_values.dtype())
                }
            }
            b.append(true);
            let mut cb = this_builder.field_builder::<StructBuilder>(4).unwrap();
            let curie_of = chunk.chunk_encoding.as_curie();
            cb.field_builder::<UInt8Builder>(0).unwrap().append_value(curie_of.cv_id);
            cb.field_builder::<UInt32Builder>(1).unwrap().append_value(curie_of.accession);
            cb.append(true);
            for (i, f) in this_schema.fields().iter().enumerate().skip(5) {
                let buf_name = BufferName::from_field(buffer_context, f.clone()).unwrap();
                let buf_name = overrides.get(&buf_name).unwrap_or(&buf_name);
                let arr = chunk.arrays.get(&buf_name.array_type).unwrap();
                let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> = this_builder.field_builder(i).unwrap();
                match array_to_arrow_type(buf_name.dtype) {
                    DataType::Int32 => {
                        let inner = b.values().as_any_mut().downcast_mut::<Int32Builder>().unwrap();
                        inner.extend(arr.to_i32()?.iter().copied().map(Some));
                    },
                    DataType::Int64 => {
                        let inner = b.values().as_any_mut().downcast_mut::<Int64Builder>().unwrap();
                        inner.extend(arr.to_i64()?.iter().copied().map(Some));
                    },
                    DataType::Float32 => {
                        let inner = b.values().as_any_mut().downcast_mut::<Float32Builder>().unwrap();
                        inner.extend(arr.to_f32()?.iter().copied().map(Some));
                    },
                    DataType::Float64 => {
                        let inner = b.values().as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                        inner.extend(arr.to_f64()?.iter().copied().map(Some));
                    },
                    DataType::LargeBinary => todo!(),
                    tp => {
                        unimplemented!("Array type {tp:?} from {:?} not supported", arr.dtype())
                    }
                }
                b.append(true);
            }
            this_builder.append(true);
        }

        Ok(this_builder.finish())
    }

    pub fn to_schema(
        &self,
        series_index_name: impl Into<String>,
        buffer_context: BufferContext,
        overrides: &HashMap<BufferName, BufferName>,
    ) -> Schema {
        let curie_fields: Vec<FieldRef> =
            serde_arrow::schema::SchemaLike::from_type::<CURIE>(Default::default()).unwrap();
        let base_name = BufferName::new(buffer_context, self.chunk_values.name.clone(), self.chunk_values.dtype());
        let base_name = overrides.get(&base_name).unwrap_or(&base_name);
        let base_name = base_name.clone().with_format(BufferFormat::Chunked);

        let field_meta = base_name.as_field_metadata();
        let mut fields_of = vec![
            Field::new(series_index_name, DataType::UInt64, true),
            Field::new(format!("{}_chunk_start", base_name), DataType::Float64, true),
            Field::new(format!("{}_chunk_end", base_name), DataType::Float64, true),
            Field::new(
                format!("{}_chunk_values", base_name),
                DataType::LargeList(Arc::new(Field::new(
                    "item",
                    array_to_arrow_type(base_name.dtype),
                    true,
                ))),
                true,
            ).with_metadata(field_meta),
            Field::new(
                "chunk_encoding",
                DataType::Struct(curie_fields.into()),
                true,
            ),
        ];
        let mut subfields = Vec::new();
        for (array_type, arr) in self.arrays.iter() {
            let name = BufferName::new(buffer_context, array_type.clone(), arr.dtype());
            let name = overrides.get(&name).cloned().unwrap_or(name).with_format(BufferFormat::ChunkedSecondary);
            let f = Field::new(
                name.to_string(),
                DataType::LargeList(
                    Field::new("item", array_to_arrow_type(name.dtype), true).into(),
                ),
                true,
            ).with_metadata(name.as_field_metadata());
            subfields.push((name, f));
        }
        subfields.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, f) in subfields {
            fields_of.push(f);
        }
        Schema::new(fields_of)
    }

    pub fn from_arrays(
        series_index: u64,
        main_axis: BufferName,
        arrays: &BinaryArrayMap,
        chunk_encoding: ChunkingStrategy,
        excluded_arrays: Option<&[BufferName]>,
    ) -> Result<Vec<Self>, ArrayRetrievalError> {
        let mut chunks = Vec::new();
        let main_axis_data: &DataArray = arrays
            .get(&main_axis.array_type)
            .ok_or_else(|| ArrayRetrievalError::NotFound(main_axis.array_type.clone()))?;

        macro_rules! chunked_delta {
            ($data:expr) => {
                let steps = chunk_every_k(&$data, NumCast::from(chunk_encoding.chunk_size()).unwrap());
                for step in steps {
                    let seg = &$data[step.start..step.end];
                    let (start, end, deltas) =
                        delta_encode(seg, &main_axis.array_type, &main_axis_data.dtype());
                    let mut parts = BinaryArrayMap::new();
                    for (k, v) in arrays.iter() {
                        if *k == main_axis.array_type
                            || excluded_arrays.is_some_and(|b| {
                                b.iter().any(|b| b.array_type == *k && b.dtype == v.dtype())
                            })
                        {
                            continue;
                        }
                        parts.add(
                            v.slice(step.start * v.dtype.size_of(), step.end * v.dtype.size_of())?,
                        );
                    }
                    chunks.push(Self::new(
                        series_index,
                        start,
                        end,
                        deltas,
                        ChunkingStrategy::Delta { chunk_size: NumCast::from(chunk_encoding.chunk_size()).unwrap() },
                        parts,
                    ));
                }
            };
        }

        match main_axis_data.dtype() {
            BinaryDataArrayType::Float64 => {
                chunked_delta!(main_axis_data.to_f64()?);
            }
            BinaryDataArrayType::Float32 => {
                chunked_delta!(main_axis_data.to_f32()?);
            }
            _ => return Err(ArrayRetrievalError::DataTypeSizeMismatch),
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod test {
    use std::io;

    use super::*;
    use mzdata::MZReader;

    #[test]
    fn test_encode() -> io::Result<()> {
        let mut reader = MZReader::open_path("small.mzML")?;
        let spec = reader.get_spectrum_by_index(0).unwrap();
        let arrays = spec.raw_arrays().unwrap();
        let target = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float32,
        );
        let chunks = ArrayChunk::from_arrays(0, target, arrays, ChunkingStrategy::Delta { chunk_size: 50.0 }, None)?;
        let schema = chunks[0].to_schema("spectrum_index", BufferContext::Spectrum, &HashMap::new());
        let arrow_arrays = ArrayChunk::to_arrays(&chunks, "spectrum_index", BufferContext::Spectrum, &schema, &HashMap::new())?;
        Ok(())
    }
}
