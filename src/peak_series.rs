use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;
use std::{borrow::Cow, fmt::Debug};

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, UInt8Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};
use mzdata::params::Unit;
use mzdata::spectrum::BinaryArrayMap;
use mzdata::{
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType, bindata::ArrayRetrievalError},
};

use mzpeaks::peak::{IonMobilityAwareCentroidPeak, IonMobilityAwareDeconvolutedPeak};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use serde::{Deserialize, Serialize};

use crate::param::{
    CURIE, curie_deserialize, curie_serialize, opt_curie_deserialize, opt_curie_serialize,
};
use crate::spectrum::AuxiliaryArray;

/// Convert `mzdata`'s [`BinaryDataArrayType`] to `arrow`'s [`DataType`]
pub fn array_to_arrow_type(dtype: BinaryDataArrayType) -> DataType {
    match dtype {
        BinaryDataArrayType::Unknown => DataType::UInt8,
        BinaryDataArrayType::Float64 => DataType::Float64,
        BinaryDataArrayType::Float32 => DataType::Float32,
        BinaryDataArrayType::Int64 => DataType::Int64,
        BinaryDataArrayType::Int32 => DataType::Int32,
        BinaryDataArrayType::ASCII => DataType::UInt8,
    }
}

/// Convert a [`BinaryArrayMap`] to a collection of `arrow`  [`FieldRef`] and [`ArrayRef`] with
/// unsupported arrays are spilled over as [`AuxiliaryArray`] instances.
pub fn array_map_to_schema_arrays_and_excess(
    context: BufferContext,
    array_map: &BinaryArrayMap,
    primary_array_len: usize,
    spectrum_index: u64,
    index_name: impl Into<String>,
    schema: &Schema,
    overrides: &HashMap<BufferName, BufferName>
) -> Result<(Fields, Vec<ArrayRef>, Vec<AuxiliaryArray>), ArrayRetrievalError> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    let mut auxiliary = Vec::new();

    fields.push(Arc::new(Field::new(index_name, DataType::UInt64, true)));

    let index_array = Arc::new(UInt64Array::from_value(spectrum_index, primary_array_len));
    arrays.push(index_array as ArrayRef);

    for (k, v) in array_map.iter() {
        let buffer_name = BufferName::new(context, k.clone(), v.dtype());
        let buffer_name = if let Some(buffer_name) = overrides.get(&buffer_name) {
            buffer_name
        } else {
            &buffer_name
        };

        let dtype = match v.dtype() {
            BinaryDataArrayType::Unknown => DataType::UInt8,
            BinaryDataArrayType::Float64 => DataType::Float64,
            BinaryDataArrayType::Float32 => DataType::Float32,
            BinaryDataArrayType::Int64 => DataType::Int64,
            BinaryDataArrayType::Int32 => DataType::Int32,
            BinaryDataArrayType::ASCII => DataType::UInt8,
        };

        if v.data_len()? != primary_array_len {
            unimplemented!("Still need to understand usage");
        }

        let name = buffer_name.to_string();


        let fieldref = Arc::new(Field::new(name, dtype, true));
        if schema.field_with_name(fieldref.name()).is_err() {
            auxiliary.push(AuxiliaryArray::from_data_array(v)?);
            continue;
        }
        fields.push(fieldref.clone());

        let array: ArrayRef = match buffer_name.dtype {
            BinaryDataArrayType::Unknown => Arc::new(UInt8Array::from(v.data.clone())),
            BinaryDataArrayType::Float64 => Arc::new(Float64Array::from(v.to_f64()?.to_vec())),
            BinaryDataArrayType::Float32 => Arc::new(Float32Array::from(v.to_f32()?.to_vec())),
            BinaryDataArrayType::Int64 => Arc::new(Int64Array::from(v.to_i64()?.to_vec())),
            BinaryDataArrayType::Int32 => Arc::new(Int32Array::from(v.to_i32()?.to_vec())),
            BinaryDataArrayType::ASCII => Arc::new(UInt8Array::from(v.data.clone())),
        };

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
    index_name: impl Into<String>,
    overrides: &HashMap<BufferName, BufferName>,
) -> Result<(Fields, Vec<ArrayRef>), ArrayRetrievalError> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    fields.push(Arc::new(Field::new(index_name, DataType::UInt64, true)));

    let index_array = Arc::new(UInt64Array::from_value(spectrum_index, primary_array_len));
    arrays.push(index_array as ArrayRef);

    for (k, v) in array_map.iter() {
        let buffer_name = BufferName::new(context, k.clone(), v.dtype());
        let buffer_name = if let Some(buffer_name) = overrides.get(&buffer_name) {
            buffer_name
        } else {
            &buffer_name
        };

        if v.data_len()? != primary_array_len {
            unimplemented!("Still need to understand usage");
        }

        let fieldref = buffer_name.to_field();
        fields.push(fieldref);

        let array: ArrayRef = match buffer_name.dtype {
            BinaryDataArrayType::Unknown => Arc::new(UInt8Array::from(v.data.clone())),
            BinaryDataArrayType::Float64 => Arc::new(Float64Array::from(v.to_f64()?.to_vec())),
            BinaryDataArrayType::Float32 => Arc::new(Float32Array::from(v.to_f32()?.to_vec())),
            BinaryDataArrayType::Int64 => Arc::new(Int64Array::from(v.to_i64()?.to_vec())),
            BinaryDataArrayType::Int32 => Arc::new(Int32Array::from(v.to_i32()?.to_vec())),
            BinaryDataArrayType::ASCII => Arc::new(UInt8Array::from(v.data.clone())),
        };

        arrays.push(array);
    }
    Ok((fields.into(), arrays))
}

/// Convert a peak list to a collection of Arrow Arrays
pub trait ToMzPeakDataSeries: Sized + BuildArrayMapFrom {
    /// Get the definition of Arrow arrays that this will be stored as for populating
    /// the schema
    fn to_fields() -> Fields;

    /// Construct a collection of Arrow arrays from the specified peak list
    fn to_arrays(spectrum_index: u64, peaks: &[Self], overrides: &HashMap<BufferName, BufferName>) -> (Fields, Vec<ArrayRef>);
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

    fn to_arrays(spectrum_index: u64, peaks: &[Self], overrides: &HashMap<BufferName, BufferName>) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            "spectrum_index",
            overrides
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

    fn to_arrays(spectrum_index: u64, peaks: &[Self], overrides: &HashMap<BufferName, BufferName>) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            "spectrum_index",
            overrides
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

    fn to_arrays(spectrum_index: u64, peaks: &[Self], overrides: &HashMap<BufferName, BufferName>) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            "spectrum_index",
            overrides
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

    fn to_arrays(spectrum_index: u64, peaks: &[Self], overrides: &HashMap<BufferName, BufferName>) -> (Fields, Vec<ArrayRef>) {
        let map = BuildArrayMapFrom::as_arrays(peaks);
        array_map_to_schema_arrays(
            BufferContext::Spectrum,
            &map,
            peaks.len(),
            spectrum_index,
            "spectrum_index",
            overrides
        )
        .unwrap()
    }
}

/// Whether an data array series is associated with a spectrum or a chromatogram
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferContext {
    Spectrum,
    Chromatogram,
}

impl BufferContext {
    pub fn index_field(&self) -> FieldRef {
        Arc::new(Field::new(
            match self {
                BufferContext::Spectrum => "spectrum_index",
                BufferContext::Chromatogram => "chromatogram_index",
            },
            DataType::UInt64,
            true,
        ))
    }
}

/// Composite structure for directly naming a data array series
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct BufferName {
    context: BufferContext,
    array_type: ArrayType,
    dtype: BinaryDataArrayType,
    unit: Unit,
}

impl PartialOrd for BufferName {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.context.partial_cmp(&other.context) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.array_type.partial_cmp(&other.array_type) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }

        match self.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "u8",
        }
        .partial_cmp(match other.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "u8",
        })
    }
}

/// Convert a [`CURIE`] into an [`ArrayType`], or return `None` if the CURIE
/// doesn't correspond to an [`ArrayType`] term.
pub fn array_type_from_accession(accession: crate::param::CURIE) -> Option<ArrayType> {
    let accession = mzdata::params::CURIE::from(accession);
    let tp = if accession == ArrayType::MZArray.as_param_const().curie()? {
        ArrayType::MZArray
    } else if accession == ArrayType::IntensityArray.as_param_const().curie()? {
        ArrayType::IntensityArray
    } else if accession == ArrayType::ChargeArray.as_param_const().curie()? {
        ArrayType::ChargeArray
    } else if accession == ArrayType::SignalToNoiseArray.as_param_const().curie()? {
        ArrayType::SignalToNoiseArray
    } else if accession == ArrayType::TimeArray.as_param_const().curie()? {
        ArrayType::TimeArray
    } else if accession == ArrayType::WavelengthArray.as_param_const().curie()? {
        ArrayType::WavelengthArray
    } else if accession == ArrayType::IonMobilityArray.as_param_const().curie()? {
        ArrayType::IonMobilityArray
    } else if accession == ArrayType::MeanIonMobilityArray.as_param_const().curie()? {
        ArrayType::MeanIonMobilityArray
    } else if accession == ArrayType::MeanDriftTimeArray.as_param_const().curie()? {
        ArrayType::MeanDriftTimeArray
    } else if accession
        == ArrayType::MeanInverseReducedIonMobilityArray
            .as_param_const()
            .curie()?
    {
        ArrayType::MeanInverseReducedIonMobilityArray
    } else if accession == ArrayType::RawIonMobilityArray.as_param_const().curie()? {
        ArrayType::RawIonMobilityArray
    } else if accession == ArrayType::RawDriftTimeArray.as_param_const().curie()? {
        ArrayType::RawDriftTimeArray
    } else if accession
        == ArrayType::RawInverseReducedIonMobilityArray
            .as_param_const()
            .curie()?
    {
        ArrayType::RawInverseReducedIonMobilityArray
    } else if accession
        == ArrayType::DeconvolutedIonMobilityArray
            .as_param_const()
            .curie()?
    {
        ArrayType::DeconvolutedIonMobilityArray
    } else if accession
        == ArrayType::DeconvolutedDriftTimeArray
            .as_param_const()
            .curie()?
    {
        ArrayType::DeconvolutedDriftTimeArray
    } else if accession
        == ArrayType::DeconvolutedInverseReducedIonMobilityArray
            .as_param_const()
            .curie()?
    {
        ArrayType::DeconvolutedInverseReducedIonMobilityArray
    } else if accession == ArrayType::BaselineArray.as_param_const().curie()? {
        ArrayType::BaselineArray
    } else if accession == ArrayType::ResolutionArray.as_param_const().curie()? {
        ArrayType::ResolutionArray
    } else if accession == ArrayType::PressureArray.as_param_const().curie()? {
        ArrayType::PressureArray
    } else if accession == ArrayType::TemperatureArray.as_param_const().curie()? {
        ArrayType::TemperatureArray
    } else if accession == ArrayType::FlowRateArray.as_param_const().curie()? {
        ArrayType::FlowRateArray
    } else if accession
        == (ArrayType::NonStandardDataArray {
            name: "".to_string().into(),
        })
        .as_param_const()
        .curie()?
    {
        ArrayType::NonStandardDataArray {
            name: "".to_string().into(),
        }
    } else {
        return None;
    };
    Some(tp)
}

/// Convert a [`CURIE`] into an [`BinaryDataArrayType`], or return `None` if the CURIE
/// doesn't correspond to an [`BinaryDataArrayType`] term.
pub fn binary_datatype_from_accession(accession: crate::CURIE) -> Option<BinaryDataArrayType> {
    let accession = accession.into();
    match accession {
        x if Some(x) == BinaryDataArrayType::Float32.curie() => Some(BinaryDataArrayType::Float32),
        x if Some(x) == BinaryDataArrayType::Float64.curie() => Some(BinaryDataArrayType::Float64),
        x if Some(x) == BinaryDataArrayType::Int32.curie() => Some(BinaryDataArrayType::Int32),
        x if Some(x) == BinaryDataArrayType::Int64.curie() => Some(BinaryDataArrayType::Int64),
        x if Some(x) == BinaryDataArrayType::ASCII.curie() => Some(BinaryDataArrayType::ASCII),
        _ => None,
    }
}

impl BufferName {
    pub const fn new(
        context: BufferContext,
        array_type: ArrayType,
        dtype: BinaryDataArrayType,
    ) -> Self {
        Self {
            context,
            array_type,
            dtype,
            unit: Unit::Unknown,
        }
    }

    pub const fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    pub fn from_field(context: BufferContext, field: FieldRef) -> Option<Self> {
        let mut array_type = None;
        let mut dtype = None;
        let mut unit = Unit::Unknown;
        let mut name = None;
        for (k, v) in field.metadata().iter() {
            match k.as_str() {
                "unit" => {
                    unit = Unit::from_accession(v);
                }
                "array_accession" => {
                    array_type = array_type_from_accession(v.parse().ok()?);
                }
                "data_type_accession" => {
                    let accession: crate::CURIE = v.parse().ok()?;
                    dtype = binary_datatype_from_accession(accession);
                }
                "array_name" => {
                    name = Some(v.to_string());
                }
                _ => {}
            }
        }

        match (array_type, dtype, name) {
            (Some(array_type), Some(dtype), Some(array_name)) => {
                let mut this = Self {
                    array_type,
                    context,
                    dtype,
                    unit,
                };
                if let ArrayType::NonStandardDataArray { name } = &mut this.array_type {
                    *name = array_name.into();
                }
                Some(this)
            }
            _ => None,
        }
    }

    pub fn to_field(&self) -> FieldRef {
        let f = Field::new(self.to_string(), array_to_arrow_type(self.dtype), true).with_metadata(
            [
                (
                    "unit".to_string(),
                    self.unit
                        .to_curie()
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "array_accession".to_string(),
                    self.array_type
                        .as_param_const()
                        .curie()
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "data_type_accession".to_string(),
                    self.dtype
                        .curie()
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "array_name".to_string(),
                    self.array_type.as_param_const().name().to_string(),
                ),
            ]
            .into_iter()
            .collect(),
        );
        Arc::new(f)
    }
}

impl Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let context = match self.context {
            BufferContext::Spectrum => "spectrum",
            BufferContext::Chromatogram => "chromatogram",
        };
        let tp_name = match &self.array_type {
            ArrayType::Unknown => Cow::Borrowed("unknown"),
            ArrayType::MZArray => Cow::Borrowed("mz"),
            ArrayType::IntensityArray => Cow::Borrowed("intensity"),
            ArrayType::ChargeArray => Cow::Borrowed("charge"),
            ArrayType::SignalToNoiseArray => Cow::Borrowed("snr"),
            ArrayType::TimeArray => Cow::Borrowed("time"),
            ArrayType::WavelengthArray => Cow::Borrowed("wavelength"),
            ArrayType::IonMobilityArray => Cow::Borrowed("ion_mobility"),
            ArrayType::MeanIonMobilityArray => Cow::Borrowed("mean_ion_mobility"),
            ArrayType::RawIonMobilityArray => Cow::Borrowed("raw_ion_mobility"),
            ArrayType::DeconvolutedIonMobilityArray => Cow::Borrowed("deconvoluted_ion_mobility"),
            ArrayType::NonStandardDataArray { name } => {
                Cow::Owned(name.replace(['/', ' ', '.'], "_"))
            }
            ArrayType::BaselineArray => Cow::Borrowed("baseline"),
            ArrayType::ResolutionArray => Cow::Borrowed("resolution"),

            ArrayType::RawInverseReducedIonMobilityArray => {
                Cow::Borrowed("raw_inverse_reduced_ion_mobility")
            }
            ArrayType::MeanInverseReducedIonMobilityArray => {
                Cow::Borrowed("mean_inverse_reduced_ion_mobility")
            }

            ArrayType::RawDriftTimeArray => Cow::Borrowed("raw_drift_time"),
            ArrayType::MeanDriftTimeArray => Cow::Borrowed("mean_drift_time"),
            _ => Cow::Owned(
                self.array_type
                    .to_string()
                    .replace(['/', ' ', '.'], "_")
                    .to_lowercase()
                    .replace("array", "_array"),
            ),
        };
        let dtype = match self.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "u8",
        };
        write!(f, "{}_{}_{}", context, tp_name, dtype)
    }
}

/// Describes an array that is encoded long-form in the data file
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayIndexEntry {
    /// Is this a spectrum or chromatogram array?
    pub context: BufferContext,
    /// The prefix to this field in the schema
    pub prefix: String,
    /// The complete path to this field from the root of the schema
    pub path: String,
    /// The name of array, either given by `array_type` or a user-defined name
    pub name: String,
    /// The kind of physical data stored in the array
    pub data_type: DataType,
    /// The kind of array being stored semantically
    pub array_type: ArrayType,
    /// The unit of the values in the array
    pub unit: Unit,
}

/// A JSON-serializable version of [`ArrayIndexEntry`].
///
/// They can be inter-converted
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SerializedArrayIndexEntry {
    pub context: String,
    pub prefix: String,
    pub path: String,
    #[serde(
        serialize_with = "curie_serialize",
        deserialize_with = "curie_deserialize"
    )]
    pub data_type: CURIE,
    #[serde(
        serialize_with = "curie_serialize",
        deserialize_with = "curie_deserialize"
    )]
    pub array_type: CURIE,
    pub array_name: String,

    #[serde(
        serialize_with = "opt_curie_serialize",
        deserialize_with = "opt_curie_deserialize"
    )]
    pub unit: Option<CURIE>,
}

/// Convert an Arrow [`DataType`] to a [`BinaryDataArrayType`]
pub(crate) const fn arrow_to_array_type(data_type: &DataType) -> Option<BinaryDataArrayType> {
    match data_type {
        DataType::UInt8 => Some(BinaryDataArrayType::ASCII),
        DataType::Int32 => Some(BinaryDataArrayType::Int32),
        DataType::Int64 => Some(BinaryDataArrayType::Int64),
        DataType::Float32 => Some(BinaryDataArrayType::Float32),
        DataType::Float64 => Some(BinaryDataArrayType::Float64),
        _ => None,
    }
}

impl From<ArrayIndexEntry> for SerializedArrayIndexEntry {
    fn from(value: ArrayIndexEntry) -> Self {
        let context = match value.context {
            BufferContext::Spectrum => "spectrum".into(),
            BufferContext::Chromatogram => "chromatogram".into(),
        };

        Self {
            context,
            prefix: value.prefix,
            path: value.path,
            data_type: match value.data_type {
                DataType::UInt8 => BinaryDataArrayType::ASCII.curie().unwrap().into(),
                DataType::Int32 => BinaryDataArrayType::Int32.curie().unwrap().into(),
                DataType::Int64 => BinaryDataArrayType::Int64.curie().unwrap().into(),
                DataType::Float32 => BinaryDataArrayType::Float32.curie().unwrap().into(),
                DataType::Float64 => BinaryDataArrayType::Float64.curie().unwrap().into(),
                _ => todo!("Cannot translate {:?} into CURIE", value.data_type),
            },
            array_type: value.array_type.as_param_const().curie().unwrap().into(),
            array_name: match &value.array_type {
                ArrayType::NonStandardDataArray { name } => name.to_string(),
                _ => value.array_type.as_param_const().name().to_string(),
            },
            unit: value.unit.to_curie().map(|c| c.into()),
        }
    }
}

impl From<SerializedArrayIndexEntry> for ArrayIndexEntry {
    fn from(value: SerializedArrayIndexEntry) -> Self {
        let context = match value.context.as_str() {
            "spectrum" => BufferContext::Spectrum,
            "chromatogram" => BufferContext::Chromatogram,
            _ => todo!("Could not infer context from {value:?}"),
        };

        Self {
            context,
            prefix: value.prefix,
            name: value
                .path
                .rsplit_once(".")
                .map(|s| s.1.to_string())
                .unwrap_or_else(|| value.path.to_string()),
            path: value.path,
            array_type: array_type_from_accession(value.array_type).unwrap_or(ArrayType::Unknown),
            data_type: array_to_arrow_type(
                binary_datatype_from_accession(value.data_type).unwrap_or_default(),
            ),
            unit: value
                .unit
                .map(|x| Unit::from_curie(&x.into()))
                .unwrap_or_default(),
        }
    }
}

impl ArrayIndexEntry {
    pub fn new(
        context: BufferContext,
        prefix: String,
        path: String,
        name: String,
        data_type: DataType,
        array_type: ArrayType,
        unit: Unit,
    ) -> Self {
        Self {
            context,
            prefix,
            path,
            name,
            data_type,
            array_type,
            unit,
        }
    }

    pub fn from_buffer_name(prefix: String, buffer_name: BufferName) -> Self {
        let path = vec![prefix.clone(), buffer_name.to_string()].join(".");
        Self {
            context: buffer_name.context,
            prefix,
            path,
            data_type: array_to_arrow_type(buffer_name.dtype),
            name: buffer_name.to_string(),
            array_type: buffer_name.array_type,
            unit: buffer_name.unit,
        }
    }
}

/// A collection of [`ArrayIndexEntry`] under a specific prefix.
///
/// Mimics a subset of [`HashMap`] API
#[derive(Debug, Default, Clone)]
pub struct ArrayIndex {
    /// The prefix to the arrays
    pub prefix: String,
    /// The collection of array index entries
    pub entries: HashMap<ArrayType, ArrayIndexEntry>,
}

impl ArrayIndex {
    pub fn new(prefix: String, entries: HashMap<ArrayType, ArrayIndexEntry>) -> Self {
        Self { prefix, entries }
    }

    pub fn get(&self, key: &ArrayType) -> Option<&ArrayIndexEntry> {
        self.entries.get(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn contains_key(&self, k: &ArrayType) -> bool {
        self.entries.contains_key(k)
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, ArrayType, ArrayIndexEntry> {
        self.entries.iter()
    }

    pub fn insert(&mut self, k: ArrayType, v: ArrayIndexEntry) -> Option<ArrayIndexEntry> {
        self.entries.insert(k, v)
    }

    /// Serialize the index to JSON as a string
    pub fn to_json(&self) -> String {
        let serialized: SerializedArrayIndex = self.clone().into();
        serde_json::to_string_pretty(&serialized).unwrap()
    }

    /// Deserialize the index from a JSON string
    pub fn from_json(text: &str) -> Self {
        let serialized: SerializedArrayIndex = serde_json::from_str(text).unwrap();
        serialized.into()
    }
}

impl From<SerializedArrayIndex> for ArrayIndex {
    fn from(value: SerializedArrayIndex) -> Self {
        let mut entries = HashMap::new();
        for v in value.entries.into_iter() {
            let v = ArrayIndexEntry::from(v);
            entries.insert(v.array_type.clone(), v);
        }

        Self {
            prefix: value.prefix,
            entries,
        }
    }
}

impl From<ArrayIndex> for SerializedArrayIndex {
    fn from(value: ArrayIndex) -> Self {
        let entries = value
            .entries
            .into_values()
            .map(SerializedArrayIndexEntry::from)
            .collect();

        Self {
            prefix: value.prefix,
            entries,
        }
    }
}

/// A serializable version of [`ArrayIndex`]
///
/// This structure is intended to be stored in the file-level metadata of
/// data array file.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SerializedArrayIndex {
    pub prefix: String,
    pub entries: Vec<SerializedArrayIndexEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferOverrideRule {
    from_buffer: BufferName,
    to_buffer: BufferName,
}