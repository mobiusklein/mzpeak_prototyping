#![allow(unused)]
use std::collections::{HashMap, HashSet};
use std::ops::AddAssign;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, AsArray, Float32Array, Float32Builder, Float64Array,
    Float64Builder, Int32Array, Int32Builder, Int64Array, Int64Builder, LargeBinaryArray,
    LargeBinaryBuilder, LargeListBuilder, StructArray, StructBuilder, UInt8Array, UInt8Builder,
    UInt32Builder, UInt64Array, UInt64Builder,
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

use crate::filter::{take_data_array, _skip_zero_runs_iter};
use crate::peak_series::{BufferContext, BufferFormat, BufferName, MZ_ARRAY, array_to_arrow_type};
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkingStrategy {
    Basic { chunk_size: f64 },
    Delta { chunk_size: f64 },
    NumpressLinear { chunk_size: f64 },
}

impl ChunkingStrategy {
    pub fn as_curie(&self) -> CURIE {
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
                let bytes = Field::new(
                    format!("{}_numpress_bytes", main_axis_name),
                    DataType::LargeBinary,
                    true,
                );
                vec![bytes]
            }
        }
    }

    pub fn encode_extra_arrow(
        &self,
        main_axis_name: &BufferName,
        chunk: &ArrayChunk,
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
                    return
                }
                visited.insert(idx);

                let b: &mut LargeBinaryBuilder = chunk_builder.field_builder(idx).unwrap();

                if matches!(chunk.chunk_encoding, Self::NumpressLinear { chunk_size: _ }) {
                    let bytes = chunk
                        .chunk_values
                        .encode_bytestring(BinaryCompressionType::NumpressLinear);

                    b.append_value(&bytes);
                } else {
                    b.append_null();
                }
            }
        }
    }

    pub fn chunk_size(&self) -> f64 {
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

    pub fn decode_arrow(&self, array: &ArrayRef, start_value: f64, accumulator: &mut DataArray) {
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
                    delta_decode(it.values(), start_value as f32, accumulator);
                }
                DataType::Float64 => {
                    let it = array.as_primitive::<Float64Type>();
                    delta_decode(it.values(), start_value, accumulator);
                }
                _ => panic!(
                    "Data type {:?} is not supported by chunk decoding",
                    array.data_type()
                ),
            },
            ChunkingStrategy::NumpressLinear { chunk_size } => match array.data_type() {
                DataType::Float64 => {
                    let it = array.as_primitive::<Float64Type>();
                    todo!("Still working on numpress decoding")
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

    pub fn to_arrow(
        chunks: &[Self],
        series_index_name: impl Into<String>,
        buffer_context: BufferContext,
        schema: &Schema,
        overrides: &HashMap<BufferName, BufferName>,
        encodings: &[ChunkingStrategy],
    ) -> Result<StructArray, ArrayRetrievalError> {
        let this_schema =
            chunks[0].to_schema(series_index_name, buffer_context, overrides, encodings);

        let main_axis = BufferName::new(
            buffer_context,
            chunks[0].chunk_values.name().clone(),
            chunks[0].chunk_values.dtype(),
        );

        let main_axis = overrides.get(&main_axis).unwrap_or(&main_axis);

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
                match array_to_arrow_type(main_axis.dtype) {
                    DataType::Int32 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Int32Builder>()
                            .unwrap();
                        inner.extend(chunk.chunk_values.to_i32()?.iter().copied().map(Some));
                    }
                    DataType::Int64 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Int64Builder>()
                            .unwrap();
                        inner.extend(chunk.chunk_values.to_i64()?.iter().copied().map(Some));
                    }
                    DataType::Float32 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Float32Builder>()
                            .unwrap();
                        inner.extend(chunk.chunk_values.to_f32()?.iter().copied().map(Some));
                    }
                    DataType::Float64 => {
                        let inner = b
                            .values()
                            .as_any_mut()
                            .downcast_mut::<Float64Builder>()
                            .unwrap();
                        inner.extend(chunk.chunk_values.to_f64()?.iter().copied().map(Some));
                    }
                    DataType::LargeBinary => todo!(),
                    tp => {
                        unimplemented!(
                            "Array type {tp:?} from {:?} not supported",
                            chunk.chunk_values.dtype()
                        )
                    }
                }
                b.append(true);
            }

            for encoding in encodings {
                encoding.encode_extra_arrow(
                    &main_axis,
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
                if let Some(buf_name) = BufferName::from_field(buffer_context, f.clone()) {
                    let buf_name = overrides.get(&buf_name).unwrap_or(&buf_name);
                    let arr = chunk.arrays.get(&buf_name.array_type).unwrap();
                    let b: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                        this_builder.field_builder(i).unwrap();
                    match array_to_arrow_type(buf_name.dtype) {
                        DataType::Int32 => {
                            let inner = b
                                .values()
                                .as_any_mut()
                                .downcast_mut::<Int32Builder>()
                                .unwrap();
                            inner.extend(arr.to_i32()?.iter().copied().map(Some));
                        }
                        DataType::Int64 => {
                            let inner = b
                                .values()
                                .as_any_mut()
                                .downcast_mut::<Int64Builder>()
                                .unwrap();
                            inner.extend(arr.to_i64()?.iter().copied().map(Some));
                        }
                        DataType::Float32 => {
                            let inner = b
                                .values()
                                .as_any_mut()
                                .downcast_mut::<Float32Builder>()
                                .unwrap();
                            inner.extend(arr.to_f32()?.iter().copied().map(Some));
                        }
                        DataType::Float64 => {
                            let inner = b
                                .values()
                                .as_any_mut()
                                .downcast_mut::<Float64Builder>()
                                .unwrap();
                            inner.extend(arr.to_f64()?.iter().copied().map(Some));
                        }
                        DataType::LargeBinary => todo!(),
                        tp => {
                            unimplemented!("Array type {tp:?} from {:?} not supported", arr.dtype())
                        }
                    }
                    b.append(true);
                } else {
                    if !visited.contains(&i) {
                        panic!("A column was not visited: {}", f.name());
                    }
                }
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
        encodings: &[ChunkingStrategy],
    ) -> Schema {
        let curie_fields: Vec<FieldRef> =
            serde_arrow::schema::SchemaLike::from_type::<CURIE>(Default::default()).unwrap();

        let base_name = BufferName::new(
            buffer_context,
            self.chunk_values.name.clone(),
            self.chunk_values.dtype(),
        );

        let base_name = overrides.get(&base_name).unwrap_or(&base_name);
        let base_name = base_name.clone().with_format(BufferFormat::Chunked);

        let field_meta = base_name.as_field_metadata();
        let mut fields_of = vec![
            Field::new(series_index_name, DataType::UInt64, true),
            Field::new(
                format!("{}_chunk_start", base_name),
                DataType::Float64,
                true,
            ),
            Field::new(format!("{}_chunk_end", base_name), DataType::Float64, true),
            Field::new(
                format!("{}_chunk_values", base_name),
                DataType::LargeList(Arc::new(Field::new(
                    "item",
                    array_to_arrow_type(base_name.dtype),
                    true,
                ))),
                true,
            )
            .with_metadata(field_meta),
            Field::new(
                "chunk_encoding",
                DataType::Struct(curie_fields.into()),
                true,
            ),
        ];
        let mut subfields = Vec::new();
        for (array_type, arr) in self.arrays.iter() {
            let name = BufferName::new(buffer_context, array_type.clone(), arr.dtype());
            let name = overrides
                .get(&name)
                .cloned()
                .unwrap_or(name)
                .with_format(BufferFormat::ChunkedSecondary);
            let f = Field::new(
                name.to_string(),
                DataType::LargeList(
                    Field::new("item", array_to_arrow_type(name.dtype), true).into(),
                ),
                true,
            )
            .with_metadata(name.as_field_metadata());
            subfields.push((name, f));
        }
        subfields.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, f) in subfields {
            fields_of.push(f);
        }
        for enc in encodings.iter() {
            fields_of.extend(enc.extra_arrays(&base_name));
        }
        Schema::new(fields_of)
    }

    pub fn from_arrays(
        series_index: u64,
        main_axis: BufferName,
        arrays: &BinaryArrayMap,
        chunk_encoding: ChunkingStrategy,
        excluded_arrays: Option<&[BufferName]>,
        drop_zero_intensity: bool,
    ) -> Result<Vec<Self>, ArrayRetrievalError> {
        let mut chunks = Vec::new();
        let mut subset_arrays = BinaryArrayMap::new();
        let arrays = if drop_zero_intensity {
            if let Ok(intensities) = arrays.intensities() {
                let kept_indices = _skip_zero_runs_iter::<Float32Type, _>(intensities.iter().map(|v| Some(*v)), intensities.len());
                let mut had_err= false;
                for (k, v) in arrays.iter() {
                    if let Ok(v) = take_data_array(v, &kept_indices) {
                        subset_arrays.add(v);
                    } else {
                        had_err = true;
                        log::error!("Failed to subset data array: {}", k);
                        break;
                    }
                }
                if had_err {
                    arrays
                } else {
                    &subset_arrays
                }
            } else {
                arrays
            }
        } else {
            arrays
        };

        let main_axis_data: &DataArray = arrays
            .get(&main_axis.array_type)
            .ok_or_else(|| ArrayRetrievalError::NotFound(main_axis.array_type.clone()))?;

        macro_rules! chunk_by {
            ($data:expr) => {
                let steps =
                    chunk_every_k(&$data, NumCast::from(chunk_encoding.chunk_size()).unwrap());
                for step in steps {
                    let seg = &$data[step.start..step.end];
                    let (start, end, deltas) =
                        chunk_encoding.encode(seg, &main_axis.array_type, &main_axis_data.dtype());
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
                        chunk_encoding,
                        parts,
                    ));
                }
            };
        }

        match main_axis_data.dtype() {
            BinaryDataArrayType::Float64 => {
                chunk_by!(main_axis_data.to_f64()?);
            }
            BinaryDataArrayType::Float32 => {
                chunk_by!(main_axis_data.to_f32()?);
            }
            _ => return Err(ArrayRetrievalError::DataTypeSizeMismatch),
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod test {
    use std::{fs, io};

    use super::*;
    use mzdata::MZReader;

    #[test]
    fn test_chunking() -> io::Result<()> {
        let mut reader = io::BufReader::new(fs::File::open("test/data/chunking_mzs.txt")?);

        let mut mzs: Vec<f64> = Vec::new();
        for line in reader.lines().flatten() {
            if line.is_empty() {
                continue;
            }
            mzs.push(line.parse().unwrap());
        }

        let intervals = chunk_every_k(&mzs, 10.0);

        let mut last = 0.0;
        for iv in intervals.iter() {
            let vs = &mzs[iv.start..iv.end];
            let term = vs.last().copied().unwrap();
            assert!((term - 1.0) > last, "{vs:?} was not more than 9 away from {last}");
            last = term;
        }

        Ok(())
    }

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
        let chunks = ArrayChunk::from_arrays(
            0,
            target,
            arrays,
            ChunkingStrategy::Delta { chunk_size: 50.0 },
            None,
            false
        )?;
        let schema = chunks[0].to_schema(
            "spectrum_index",
            BufferContext::Spectrum,
            &HashMap::new(),
            &[ChunkingStrategy::Delta { chunk_size: 50.0 }],
        );
        let arrow_arrays = ArrayChunk::to_arrow(
            &chunks,
            "spectrum_index",
            BufferContext::Spectrum,
            &schema,
            &HashMap::new(),
            &[ChunkingStrategy::Delta { chunk_size: 50.0 }],
        )?;
        Ok(())
    }
}
