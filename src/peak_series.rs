use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, LargeBinaryArray, UInt8Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Fields};
use mzdata::params::Unit;
use mzdata::spectrum::{BinaryArrayMap, DataArray};
use mzdata::{
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType, bindata::ArrayRetrievalError},
};

use mzpeaks::peak::{IonMobilityAwareCentroidPeak, IonMobilityAwareDeconvolutedPeak};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};

use crate::spectrum::AuxiliaryArray;

pub use crate::buffer_descriptors::{
    ArrayIndex, ArrayIndexEntry, BufferContext, BufferFormat, BufferName, BufferOverrideRule,
    SerializedArrayIndex, SerializedArrayIndexEntry, array_priority, array_type_from_accession,
    binary_datatype_from_accession,
};

pub fn ascii_array(data_array: &DataArray) -> LargeBinaryArray {
    let tokens = data_array.data.split(|s| *s == b'\0');
    let ba = LargeBinaryArray::from_iter_values(tokens);
    ba
}

pub fn data_array_to_arrow_array(
    buffer_name: &BufferName,
    data_array: &DataArray,
) -> Result<ArrayRef, ArrayRetrievalError> {
    let array: ArrayRef = match buffer_name.dtype {
        BinaryDataArrayType::Unknown => Arc::new(UInt8Array::from(data_array.data.clone())),
        BinaryDataArrayType::Float64 => Arc::new(Float64Array::from(data_array.to_f64()?.to_vec())),
        BinaryDataArrayType::Float32 => Arc::new(Float32Array::from(data_array.to_f32()?.to_vec())),
        BinaryDataArrayType::Int64 => Arc::new(Int64Array::from(data_array.to_i64()?.to_vec())),
        BinaryDataArrayType::Int32 => Arc::new(Int32Array::from(data_array.to_i32()?.to_vec())),
        BinaryDataArrayType::ASCII => Arc::new(ascii_array(&data_array)),
    };
    Ok(array)
}

/// Convert `mzdata`'s [`BinaryDataArrayType`] to `arrow`'s [`DataType`]
pub fn array_to_arrow_type(dtype: BinaryDataArrayType) -> DataType {
    match dtype {
        BinaryDataArrayType::Unknown => DataType::UInt8,
        BinaryDataArrayType::Float64 => DataType::Float64,
        BinaryDataArrayType::Float32 => DataType::Float32,
        BinaryDataArrayType::Int64 => DataType::Int64,
        BinaryDataArrayType::Int32 => DataType::Int32,
        BinaryDataArrayType::ASCII => DataType::LargeBinary,
    }
}

/// Convert a [`BinaryArrayMap`] to a collection of `arrow`  [`FieldRef`] and [`ArrayRef`] with
/// arrays not covered by `schema` are spilled over as [`AuxiliaryArray`] instances.
pub fn array_map_to_schema_arrays_and_excess(
    context: BufferContext,
    array_map: &BinaryArrayMap,
    primary_array_len: usize,
    spectrum_index: u64,
    spectrum_time: Option<f32>,
    schema: Option<&Fields>,
    overrides: &HashMap<BufferName, BufferName>,
) -> Result<(Fields, Vec<ArrayRef>, Vec<AuxiliaryArray>), ArrayRetrievalError> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    let mut auxiliary = Vec::new();

    fields.push(context.index_field());
    let index_array = Arc::new(UInt64Array::from_value(spectrum_index, primary_array_len));
    arrays.push(index_array as ArrayRef);
    if let Some(spectrum_time) = spectrum_time {
        fields.push(context.time_field());
        arrays.push(Arc::new(Float32Array::from_value(
            spectrum_time,
            primary_array_len,
        )));
    }

    for (_, v) in array_map.iter() {
        let buffer_name = BufferName::from_data_array(context, v);
        let buffer_name = if let Some(buffer_name) = overrides.get(&buffer_name) {
            buffer_name
        } else {
            &buffer_name
        };

        let fieldref = buffer_name.to_field();
        if let Some(schema) = schema {
            if schema
                .iter()
                .find(|c| c.name() == fieldref.name())
                .is_none()
            {
                log::trace!("{fieldref:?} did not map to schema {schema:?}");
                auxiliary.push(AuxiliaryArray::from_data_array(v)?);
                continue;
            }
        }

        if v.data_len()? != primary_array_len {
            if primary_array_len == 0 {
                auxiliary.push(AuxiliaryArray::from_data_array(v)?);
                continue;
            } else {
                unimplemented!(
                    "Still need to understand usage for uneven arrays: {} had {} points but primary length was {}",
                    buffer_name,
                    v.data_len()?,
                    primary_array_len
                );
            }
        }

        fields.push(fieldref.clone());

        let array: ArrayRef = data_array_to_arrow_array(buffer_name, v)?;

        arrays.push(array);
    }
    Ok((fields.into(), arrays, auxiliary))
}

/// Convert a [`BinaryArrayMap`] to a collection of `arrow`  [`FieldRef`] and [`ArrayRef`].
pub fn array_map_to_schema_arrays(
    context: BufferContext,
    array_map: &BinaryArrayMap,
    primary_array_len: usize,
    spectrum_index: u64,
    spectrum_time: Option<f32>,
    overrides: &HashMap<BufferName, BufferName>,
) -> Result<(Fields, Vec<ArrayRef>), ArrayRetrievalError> {
    let (fields, arrays, _aux) = array_map_to_schema_arrays_and_excess(
        context,
        array_map,
        primary_array_len,
        spectrum_index,
        spectrum_time,
        None,
        overrides,
    )?;
    return Ok((fields, arrays));
}

/// Convert a peak list to a collection of Arrow Arrays
pub trait ToMzPeakDataSeries: Sized + BuildArrayMapFrom {
    /// Get the definition of Arrow arrays that this will be stored as for populating
    /// the schema
    fn to_fields() -> Fields;

    /// Construct a collection of Arrow arrays from the specified peak list
    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &HashMap<BufferName, BufferName>,
    ) -> (Fields, Vec<ArrayRef>);
}

pub const MZ_ARRAY: BufferName = BufferName::new(
    BufferContext::Spectrum,
    ArrayType::MZArray,
    BinaryDataArrayType::Float64,
)
.with_unit(Unit::MZ);
pub const INTENSITY_ARRAY: BufferName = BufferName::new(
    BufferContext::Spectrum,
    ArrayType::IntensityArray,
    BinaryDataArrayType::Float32,
)
.with_unit(Unit::DetectorCounts);
pub const CHARGE_ARRAY: BufferName = BufferName::new(
    BufferContext::Spectrum,
    ArrayType::ChargeArray,
    BinaryDataArrayType::Int32,
);

impl ToMzPeakDataSeries for CentroidPeak {
    fn to_fields() -> Fields {
        vec![
            BufferContext::Spectrum.index_field(),
            MZ_ARRAY.to_field(),
            INTENSITY_ARRAY.to_field(),
        ]
        .into()
    }

    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &HashMap<BufferName, BufferName>,
    ) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            spectrum_time,
            overrides,
        )
        .unwrap()
    }
}

impl ToMzPeakDataSeries for IonMobilityAwareCentroidPeak {
    fn to_fields() -> Fields {
        vec![
            BufferContext::Spectrum.index_field(),
            MZ_ARRAY.to_field(),
            INTENSITY_ARRAY.to_field(),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IonMobilityArray,
                BinaryDataArrayType::Float64,
            )
            .to_field(),
        ]
        .into()
    }

    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &HashMap<BufferName, BufferName>,
    ) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            spectrum_time,
            overrides,
        )
        .unwrap()
    }
}

impl ToMzPeakDataSeries for DeconvolutedPeak {
    fn to_fields() -> Fields {
        vec![
            BufferContext::Spectrum.index_field(),
            MZ_ARRAY.to_field(),
            INTENSITY_ARRAY.to_field(),
            CHARGE_ARRAY.to_field(),
        ]
        .into()
    }

    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &HashMap<BufferName, BufferName>,
    ) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            spectrum_time,
            overrides,
        )
        .unwrap()
    }
}

impl ToMzPeakDataSeries for IonMobilityAwareDeconvolutedPeak {
    fn to_fields() -> Fields {
        vec![
            BufferContext::Spectrum.index_field(),
            MZ_ARRAY.to_field(),
            INTENSITY_ARRAY.to_field(),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IonMobilityArray,
                BinaryDataArrayType::Float64,
            )
            .to_field(),
            CHARGE_ARRAY.to_field(),
        ]
        .into()
    }

    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &HashMap<BufferName, BufferName>,
    ) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            spectrum_time,
            overrides,
        )
        .unwrap()
    }
}
