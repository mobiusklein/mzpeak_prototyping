use std::{borrow::Cow, collections::HashMap, fmt::Display, str::FromStr, sync::Arc};

use arrow::datatypes::{DataType, Field, FieldRef};
use mzdata::{
    params::{ParamLike, Unit},
    prelude::ByteArrayView,
    spectrum::{ArrayType, BinaryDataArrayType, DataArray, bindata::BinaryCompressionType},
};
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};

use crate::param::{
    CURIE, curie_deserialize, curie_serialize, opt_curie_deserialize, opt_curie_serialize,
};
use crate::peak_series::array_to_arrow_type;

/// Whether an data array series is associated with a spectrum or a chromatogram
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferContext {
    Spectrum,
    Chromatogram,
}

impl BufferContext {
    pub const fn index_name(&self) -> &'static str {
        match self {
            BufferContext::Spectrum => "spectrum_index",
            BufferContext::Chromatogram => "chromatogram_index",
        }
    }

    pub fn index_field(&self) -> FieldRef {
        Arc::new(Field::new(self.index_name(), DataType::UInt64, true))
    }

    pub const fn time_name(&self) -> &'static str {
        match self {
            BufferContext::Spectrum => "spectrum_time",
            BufferContext::Chromatogram => "chromatogram_time",
        }
    }

    pub fn time_field(&self) -> FieldRef {
        Arc::new(Field::new(self.time_name(), DataType::Float32, true))
    }

    pub const fn name(&self) -> &'static str {
        match self {
            BufferContext::Spectrum => "spectrum",
            BufferContext::Chromatogram => "chromatogram",
        }
    }

    pub const fn default_sorted_array(&self) -> ArrayType {
        match self {
            BufferContext::Spectrum => ArrayType::MZArray,
            BufferContext::Chromatogram => ArrayType::TimeArray,
        }
    }
}

impl Display for BufferContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The layout of a buffer denoting the shape of the data in each position in the buffer.
///
/// This is part of a [`BufferName`] and helps guide a reader in decoding signal data.
///
/// ## Note
/// This enum derives a [`Ord`], which means that the order of the names matters. This
/// is tied to how values are explicitly ordered elsewhere in the source code.
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferFormat {
    /// A series of contiguous points
    #[default]
    Point,
    /// The start value of the encoded values of a chunk, to be visible to the indexer.
    /// Goes with [`BufferFormat::Chunked`]
    ChunkBoundsStart,
    /// The end value of the encoded values of a chunk, to be visible to the indexer.
    /// Goes with [`BufferFormat::Chunked`]
    ChunkBoundsEnd,
    /// A contiguous list of values in a chunk that may be transformed. It will have a start
    /// and end value encoded in parallel with it, along with an encoding to tell the reader
    /// how to reconstruct the original values.
    Chunked,
    /// The CURIE defining how the chunk was encoded. Goes with [`BufferFormat::Chunked`]
    ChunkEncoding,
    /// A contiguous list of values in a chunk contiguous with a [`BufferFormat::Chunked`] array
    ChunkedSecondary,
    ChunkedTransform
}

impl BufferFormat {
    /// Get the prefix suggested for this format type
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Chunked |
            Self::ChunkedSecondary |
            Self::ChunkBoundsStart |
            Self::ChunkBoundsEnd |
            Self::ChunkEncoding |
            Self::ChunkedTransform => "chunk",
        }
    }
}

impl FromStr for BufferFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "point" => Ok(Self::Point),
            "chunk_values" => Ok(Self::Chunked),
            "secondary_chunk" => Ok(Self::ChunkedSecondary),
            "chunk_start" => Ok(Self::ChunkBoundsStart),
            "chunk_end" => Ok(Self::ChunkBoundsEnd),
            "chunk_encoding" => Ok(Self::ChunkEncoding),
            "chunk_transform" => Ok(Self::ChunkedTransform),
            _ => Err(format!("{s} not recognized as a buffer format")),
        }
    }
}

impl Display for BufferFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferFormat::Point => f.write_str("point"),
            BufferFormat::Chunked => f.write_str("chunk_values"),
            BufferFormat::ChunkedSecondary => f.write_str("secondary_chunk"),
            BufferFormat::ChunkBoundsStart => f.write_str("chunk_start"),
            BufferFormat::ChunkBoundsEnd => f.write_str("chunk_end"),
            BufferFormat::ChunkEncoding => f.write_str("chunk_encoding"),
            BufferFormat::ChunkedTransform => f.write_str("chunk_transform")
        }
    }
}

impl PartialEq<str> for BufferFormat {
    fn eq(&self, other: &str) -> bool {
        self.to_string().eq_ignore_ascii_case(other)
    }
}

/// Whether or not a [`BufferName`] (or equivalent) is the main instance of [`BufferName::array_type`]
/// or not. This lets us reliably simplify the names of some arrays to be easy to recognize without
/// metadata parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay, DeserializeFromStr, Hash)]
pub enum BufferPriority {
    /// This is the primary array of this type, it receives the short naming scheme, along with the
    /// regular metadata in the [`ArrayIndex`].
    Primary,
    /// This is a secondary array of this type, it will receive an implementation-defined name, but
    /// will still be annotated in the [`ArrayIndex`].
    Secondary,
}

impl Display for BufferPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferPriority::Primary => f.write_str("primary"),
            BufferPriority::Secondary => f.write_str("secondary"),
        }
    }
}

/// Greater priority translates to a greater than relationship, i.e. `Primary > Secondary`
/// for sorting convenience.
impl Ord for BufferPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if matches!(self, Self::Primary) {
            if self != other {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        } else if matches!(self, Self::Secondary) {
            if matches!(other, Self::Primary) {
                std::cmp::Ordering::Less
            } else if self != other {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl PartialOrd for BufferPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FromStr for BufferPriority {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "primary" => Ok(Self::Primary),
            "secondary" => Ok(Self::Secondary),
            _ => Err(format!("Unknown buffer priority {s}")),
        }
    }
}

/// Composite structure for directly naming a data array series and describing
/// its metadata, e.g. data type, array type + unit, format (layout in points or chunks)
/// and any other "stuff" we might need.
///
/// This type also serializes to a JSON-friendly key-value pair map that is also used as
/// [`Field`] metadata to make runtime inspection possible
#[derive(Clone, Debug, Eq)]
pub struct BufferName {
    /// Is this a spectrum or chromatogram array?
    pub context: BufferContext,
    /// The kind of array being stored semantically
    pub array_type: ArrayType,
    /// The kind of physical data stored in the array
    pub dtype: BinaryDataArrayType,
    /// The unit of the values in the array
    pub unit: Unit,
    /// The layout of buffer, either point or chunks
    pub buffer_format: BufferFormat,
    /// Any transformations applied to this array
    pub transform: Option<BufferTransform>,
    /// The default data processing method's ID applied to this array. Alternatives may
    /// be specified in the metadata table.
    pub data_processing_id: Option<Box<str>>,
    /// Whether this is the primary array for this quantity, or a secondary array.
    /// Primary arrays get a much more succinct standardized naming scheme while
    /// all other arrays' names are implementation details.
    pub buffer_priority: Option<BufferPriority>,
    /// In what rank order this array's values are sorted, if at all, within a parent
    /// entity. This may be useful when deciding how to prioritize applying filters.
    ///
    /// Lower values are sorted first.
    pub sorting_rank: Option<u32>,
}

impl std::hash::Hash for BufferName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.context.hash(state);
        self.array_type.hash(state);
        self.dtype.hash(state);
        self.unit.hash(state);
        self.buffer_format.hash(state);
        self.transform.hash(state);
    }
}

impl PartialEq for BufferName {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context
            && self.array_type == other.array_type
            && self.dtype == other.dtype
            && self.unit == other.unit
            && self.buffer_format == other.buffer_format
    }
}

impl Ord for BufferName {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Buffers from different contexts in the same schema should be separated
        match self.context.cmp(&other.context) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        /*
        Buffers in different formats should be ordered separately,
        with chunked formats internally sub-divided by disposition.
        */
        match self.buffer_format.cmp(&other.buffer_format) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        // Different arrays should occur in a certain order
        match array_priority(&self.array_type).cmp(&array_priority(&other.array_type)) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        match self
            .buffer_priority
            .unwrap_or(BufferPriority::Secondary)
            .cmp(&other.buffer_priority.unwrap_or(BufferPriority::Secondary))
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }

        match self.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "ascii",
        }
        .cmp(match other.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "ascii",
        })
    }
}

impl PartialOrd for BufferName {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Potentially opaque transforms that need additional buffers to store the transformed data
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferTransform {
    NumpressLinear,
    NumpressSLOF,
    NumpressPIC,
    NullInterpolate,
    NullZero,
}

const NULL_INTERPOLATE: CURIE = mzdata::curie!(MS:1003901);
const NULL_ZERO: CURIE = mzdata::curie!(MS:1003902);

impl BufferTransform {
    pub fn from_curie(accession: crate::param::CURIE) -> Option<Self> {
        match accession {
            x if x == Self::NumpressSLOF.curie() => Some(Self::NumpressSLOF),
            x if x == Self::NumpressPIC.curie() => Some(Self::NumpressPIC),
            x if x == Self::NumpressLinear.curie() => Some(Self::NumpressLinear),
            x if x == NULL_INTERPOLATE => Some(Self::NullInterpolate),
            x if x == NULL_ZERO => Some(Self::NullZero),
            _ => None,
        }
    }

    pub const fn array_name_fragment(&self) -> Option<&'static str> {
        match self {
            BufferTransform::NumpressLinear => Some("numpress_linear"),
            BufferTransform::NumpressSLOF => Some("numpress_slof"),
            BufferTransform::NumpressPIC => Some("numpress_pic"),
            BufferTransform::NullInterpolate => None,
            BufferTransform::NullZero => None,
        }
    }

    pub fn curie(&self) -> CURIE {
        match self {
            BufferTransform::NumpressSLOF => BinaryCompressionType::NumpressSLOF
                .as_param()
                .unwrap()
                .curie()
                .unwrap(),
            BufferTransform::NumpressPIC => BinaryCompressionType::NumpressPIC
                .as_param()
                .unwrap()
                .curie()
                .unwrap(),
            BufferTransform::NumpressLinear => BinaryCompressionType::NumpressLinear
                .as_param()
                .unwrap()
                .curie()
                .unwrap(),
            BufferTransform::NullInterpolate => NULL_INTERPOLATE,
            BufferTransform::NullZero => NULL_ZERO,
        }
    }
}

/// Convert a [`CURIE`] into an [`ArrayType`], or return `None` if the CURIE
/// doesn't correspond to an [`ArrayType`] term.
pub fn array_type_from_accession(accession: crate::param::CURIE) -> Option<ArrayType> {
    ArrayType::from_accession(accession)
}

#[inline(always)]
/// Convert a [`CURIE`] into an [`BinaryDataArrayType`], or return `None` if the CURIE
/// doesn't correspond to an [`BinaryDataArrayType`] term.
pub fn binary_datatype_from_accession(
    accession: crate::param::CURIE,
) -> Option<BinaryDataArrayType> {
    BinaryDataArrayType::from_accession(accession)
}

/// Compute an ordering constant for [`mzdata::spectrum::ArrayType`]
pub const fn array_priority(array_type: &ArrayType) -> u64 {
    match array_type {
        ArrayType::MZArray => 1,
        ArrayType::TimeArray => 2,
        ArrayType::IntensityArray => 5,
        ArrayType::ChargeArray => 6,
        ArrayType::SignalToNoiseArray => 7,
        ArrayType::WavelengthArray => 8,
        ArrayType::IonMobilityArray => 9,
        ArrayType::MeanIonMobilityArray => 10,
        ArrayType::MeanDriftTimeArray => 11,
        ArrayType::MeanInverseReducedIonMobilityArray => 12,
        ArrayType::RawIonMobilityArray => 13,
        ArrayType::RawDriftTimeArray => 14,
        ArrayType::RawInverseReducedIonMobilityArray => 15,
        ArrayType::DeconvolutedIonMobilityArray => 16,
        ArrayType::DeconvolutedDriftTimeArray => 17,
        ArrayType::DeconvolutedInverseReducedIonMobilityArray => 18,
        ArrayType::BaselineArray => 19,
        ArrayType::ResolutionArray => 20,
        ArrayType::PressureArray => 21,
        ArrayType::TemperatureArray => 22,
        ArrayType::FlowRateArray => 22,
        ArrayType::NonStandardDataArray { name } => {
            let b = name.as_bytes();
            let n = b.len();
            let mut i: usize = 0;
            let mut k: u64 = 0;
            while i < n {
                k = k.saturating_add(b[i] as u64 * (i as u64 + 1));
                i += 1;
            }
            22u64.saturating_add(k).saturating_add(n as u64)
        }
        ArrayType::Unknown => u64::MAX - 10,
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
            buffer_format: BufferFormat::Point,
            transform: None,
            data_processing_id: None,
            buffer_priority: None,
            sorting_rank: None,
        }
    }

    pub const fn new_with_buffer_format(
        context: BufferContext,
        array_type: ArrayType,
        dtype: BinaryDataArrayType,
        buffer_format: BufferFormat,
    ) -> Self {
        Self {
            context,
            array_type,
            dtype,
            unit: Unit::Unknown,
            buffer_format,
            transform: None,
            data_processing_id: None,
            buffer_priority: None,
            sorting_rank: None,
        }
    }

    pub const fn with_priority(mut self, buffer_priority: Option<BufferPriority>) -> Self {
        self.buffer_priority = buffer_priority;
        self
    }

    pub const fn with_sorting_rank(mut self, sorting_rank: Option<u32>) -> Self {
        self.sorting_rank = sorting_rank;
        self
    }

    pub const fn with_format(mut self, buffer_format: BufferFormat) -> Self {
        self.buffer_format = buffer_format;
        self
    }

    pub const fn with_transform(mut self, transform: Option<BufferTransform>) -> Self {
        self.transform = transform;
        self
    }

    pub fn as_data_array(&self, size: usize) -> DataArray {
        let mut da = DataArray::from_name_type_size(
            &self.array_type,
            self.dtype,
            size * self.dtype.size_of(),
        );
        da.unit = self.unit;
        da
    }

    pub const fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    pub fn array_name(&self) -> String {
        if let ArrayType::NonStandardDataArray { name } = &self.array_type {
            name.to_string()
        } else {
            self.array_type.as_param(None).name().to_string()
        }
    }

    pub fn as_field_metadata(&self) -> HashMap<String, String> {
        let mut meta: HashMap<String, String> = [
            ("context".to_string(), self.context.to_string()),
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
                    .as_param(None)
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
            ("array_name".to_string(), self.array_name()),
            ("buffer_format".to_string(), self.buffer_format.to_string()),
        ]
        .into_iter()
        .collect();
        if let Some(trfm) = self.transform.as_ref() {
            meta.insert("transform".to_string(), trfm.curie().to_string());
        }
        if let Some(dp_id) = self.data_processing_id.as_ref() {
            meta.insert("data_processing_id".to_string(), dp_id.to_string());
        }
        if let Some(priority) = self.buffer_priority {
            meta.insert("buffer_priority".to_string(), priority.to_string());
        }
        if let Some(sorting_rank) = self.sorting_rank {
            meta.insert("sorting_rank".to_string(), sorting_rank.to_string());
        }
        meta
    }

    pub fn from_field(context: BufferContext, field: FieldRef) -> Option<Self> {
        let mut array_type = None;
        let mut dtype = None;
        let mut unit = Unit::Unknown;
        let mut name = None;
        let mut buffer_format = BufferFormat::Point;
        let mut transform = None;
        let mut data_processing_id = None;
        let mut buffer_priority: Option<BufferPriority> = None;
        let mut sorting_rank: Option<u32> = None;
        for (k, v) in field.metadata().iter() {
            match k.as_str() {
                "context" => {}
                "unit" => {
                    unit = Unit::from_accession(v);
                }
                "array_accession" => {
                    array_type = array_type_from_accession(
                        v.parse()
                            .inspect_err(|e| {
                                log::error!("Failed to parse array type accession: {e}")
                            })
                            .ok()?,
                    );
                }
                "data_type_accession" => {
                    let accession: crate::CURIE = v
                        .parse()
                        .inspect_err(|e| log::error!("Failed to parse data type accession: {e}"))
                        .ok()?;
                    dtype = binary_datatype_from_accession(accession);
                }
                "array_name" => {
                    name = Some(v.to_string());
                }
                "buffer_format" => {
                    buffer_format = v
                        .parse()
                        .inspect_err(|e| log::error!("Failed to parse buffer format: {e}"))
                        .ok()?;
                }
                "transform" => {
                    transform = v
                        .parse::<CURIE>()
                        .inspect_err(|e| log::error!("Failed to parse transform: {e}"))
                        .ok()
                        .and_then(BufferTransform::from_curie);
                }
                "data_processing_id" => data_processing_id = Some(v.clone().into_boxed_str()),
                "buffer_priority" => {
                    buffer_priority = v
                        .parse()
                        .inspect_err(|e| log::error!("Failed to parse buffer priority: {e}"))
                        .ok()
                }
                "sorting_rank" => {
                    sorting_rank = v
                        .parse()
                        .inspect_err(|e| log::error!("Failed to parse sorting rank: {e}"))
                        .ok()
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
                    buffer_format,
                    transform,
                    data_processing_id,
                    buffer_priority,
                    sorting_rank,
                };
                if let ArrayType::NonStandardDataArray { name } = &mut this.array_type {
                    *name = array_name.into();
                }
                Some(this)
            }
            _ => None,
        }
    }

    pub fn from_data_array(context: BufferContext, data_array: &DataArray) -> Self {
        let name = Self::new(context, data_array.name.clone(), data_array.dtype());
        name.with_unit(data_array.unit)
    }

    pub fn make_bounds_fields(&self) -> Option<(FieldRef, FieldRef)> {
        if !matches!(self.buffer_format, BufferFormat::Chunked) {
            return None;
        }
        let start = self
            .clone()
            .with_format(BufferFormat::ChunkBoundsStart)
            .to_field();
        let end = self
            .clone()
            .with_format(BufferFormat::ChunkBoundsEnd)
            .to_field();
        Some((start, end))
    }

    /// Create an Arrow field for this buffer name
    pub fn to_field(&self) -> FieldRef {
        let f = Field::new(self.to_string(), array_to_arrow_type(self.dtype), true)
            .with_metadata(self.as_field_metadata());
        Arc::new(f)
    }

    /// Update [`FieldRef`] metadata, creating a new field and returning it
    pub fn update_field(&self, field: FieldRef) -> FieldRef {
        let metadata = self.as_field_metadata();
        Arc::new(field.as_ref().clone().with_metadata(metadata))
    }
}

impl Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tp_name = match &self.array_type {
            ArrayType::Unknown => Cow::Borrowed("unknown"),
            ArrayType::MZArray => Cow::Borrowed("mz"),
            ArrayType::IntensityArray => Cow::Borrowed("intensity"),
            ArrayType::ChargeArray => Cow::Borrowed("charge"),
            ArrayType::SignalToNoiseArray => Cow::Borrowed("signal_to_noise"),
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
        if self.buffer_priority == Some(BufferPriority::Primary) {
            return match self.buffer_format {
                BufferFormat::Point | BufferFormat::ChunkedSecondary => {
                    write!(f, "{tp_name}")
                }
                BufferFormat::Chunked => write!(f, "{tp_name}_chunk_values"),
                BufferFormat::ChunkBoundsStart => write!(f, "{tp_name}_chunk_start"),
                BufferFormat::ChunkBoundsEnd => write!(f, "{tp_name}_chunk_end"),
                BufferFormat::ChunkEncoding => f.write_str("chunk_encoding"),
                BufferFormat::ChunkedTransform => {
                    if let Some(tfm) = self.transform {
                        if let Some(fragment) = tfm.array_name_fragment() {
                            write!(f, "{tp_name}_{fragment}_bytes")
                        } else {
                            panic!("Cannot create an array of `ChunkedTransform` with a transform that does not have an array name fragment");
                        }
                    } else {
                        panic!("Cannot create an array of `ChunkedTransform` without a transform");
                    }
                },
            };
        }
        let dtype = match self.dtype {
            BinaryDataArrayType::Unknown => "unknown",
            BinaryDataArrayType::Float64 => "f64",
            BinaryDataArrayType::Float32 => "f32",
            BinaryDataArrayType::Int64 => "i64",
            BinaryDataArrayType::Int32 => "i32",
            BinaryDataArrayType::ASCII => "ascii",
        };
        if matches!(self.unit, Unit::Unknown) {
            if let BufferFormat::ChunkedTransform = self.buffer_format {
                if let Some(tfm) = self.transform {
                    if let Some(fragment) = tfm.array_name_fragment() {
                        write!(f, "{tp_name}_{dtype}_{fragment}_bytes")
                    } else {
                        panic!("Cannot create an array of `ChunkedTransform` with a transform that does not have an array name fragment");
                    }
                } else {
                    panic!("Cannot create an array of `ChunkedTransform` without a transform");
                }
            } else {
                write!(f, "{tp_name}_{dtype}")
            }
        } else {
            let unit = match self.unit {
                Unit::Unknown => "",
                Unit::MZ => "mz",
                Unit::Mass => "da",
                Unit::PartsPerMillion => "ppm",
                Unit::Nanometer => "nm",
                Unit::Minute => "min",
                Unit::Second => "sec",
                Unit::Millisecond => "msec",
                Unit::VoltSecondPerSquareCentimeter => "vspc",
                Unit::DetectorCounts => "dc",
                Unit::PercentBasePeak => "bp",
                Unit::PercentBasePeakTimes100 => "bpp",
                Unit::AbsorbanceUnit => "au",
                Unit::CountsPerSecond => "cps",
                Unit::Electronvolt => "ev",
                Unit::Volt => "v",
                Unit::Celsius => "c",
                Unit::Kelvin => "k",
                Unit::Pascal => "pa",
                Unit::Psi => "psi",
                Unit::MicrolitersPerMinute => "mlpmin",
                Unit::Percent => "pct",
                Unit::Dimensionless => "",
            };
            if let BufferFormat::ChunkedTransform = self.buffer_format {
                if let Some(tfm) = self.transform {
                    if let Some(fragment) = tfm.array_name_fragment() {
                        write!(f, "{tp_name}_{dtype}_{unit}_{fragment}_bytes")
                    } else {
                        panic!("Cannot create an array of `ChunkedTransform` with a transform that does not have an array name fragment");
                    }
                } else {
                    panic!("Cannot create an array of `ChunkedTransform` without a transform");
                }
            } else {
                write!(f, "{tp_name}_{dtype}_{unit}")
            }
        }
    }
}

/// A JSON-serializable version of [`ArrayIndexEntry`].
///
/// They can be inter-converted. See [`ArrayIndexEntry`] for
/// an explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedArrayIndexEntry {
    pub context: String,
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
    #[serde(default)]
    pub buffer_format: String,
    #[serde(
        serialize_with = "opt_curie_serialize",
        deserialize_with = "opt_curie_deserialize",
        default
    )]
    pub transform: Option<CURIE>,
    pub data_processing_id: Option<Box<str>>,
    pub buffer_priority: Option<BufferPriority>,
    pub sorting_rank: Option<u32>,
}

/// Convert an Arrow [`DataType`] to a [`BinaryDataArrayType`]
pub(crate) const fn arrow_to_array_type(data_type: &DataType) -> Option<BinaryDataArrayType> {
    match data_type {
        DataType::LargeBinary => Some(BinaryDataArrayType::ASCII),
        DataType::Int32 => Some(BinaryDataArrayType::Int32),
        DataType::Int64 => Some(BinaryDataArrayType::Int64),
        DataType::UInt32 => Some(BinaryDataArrayType::Int32),
        DataType::UInt64 => Some(BinaryDataArrayType::Int64),
        DataType::Float32 => Some(BinaryDataArrayType::Float32),
        DataType::Float64 => Some(BinaryDataArrayType::Float64),
        _ => None,
    }
}

impl From<ArrayIndexEntry> for SerializedArrayIndexEntry {
    fn from(value: ArrayIndexEntry) -> Self {
        let context = value.context.name().to_string();

        Self {
            context,
            path: value.path,
            data_type: match value.data_type {
                DataType::LargeBinary => BinaryDataArrayType::ASCII.curie().unwrap(),
                DataType::Int32 => BinaryDataArrayType::Int32.curie().unwrap(),
                DataType::Int64 => BinaryDataArrayType::Int64.curie().unwrap(),
                DataType::Float32 => BinaryDataArrayType::Float32.curie().unwrap(),
                DataType::Float64 => BinaryDataArrayType::Float64.curie().unwrap(),
                _ => todo!("Cannot translate {:?} into CURIE", value.data_type),
            },
            array_type: value.array_type.as_param(None).curie().unwrap(),
            array_name: match &value.array_type {
                ArrayType::NonStandardDataArray { name } => name.to_string(),
                _ => value.array_type.as_param_const().name().to_string(),
            },
            unit: value.unit.to_curie(),
            buffer_format: value.buffer_format.to_string(),
            transform: value.transform.map(|t| t.curie()),
            data_processing_id: value.data_processing_id,
            buffer_priority: value.buffer_priority,
            sorting_rank: value.sorting_rank,
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
            unit: value.unit.map(|x| Unit::from_curie(&x)).unwrap_or_default(),
            buffer_format: value
                .buffer_format
                .parse::<BufferFormat>()
                .unwrap_or(BufferFormat::Point),
            transform: value.transform.and_then(|t| {
                BufferTransform::from_curie(t).or_else(|| {
                    log::warn!("Failed to translate {t} into a buffer transform");
                    None
                })
            }),
            data_processing_id: value.data_processing_id,
            buffer_priority: value.buffer_priority,
            sorting_rank: value.sorting_rank,
        }
    }
}

/// Describes an array that is encoded long-form in the data file
///
/// This type is a logical extension [`BufferName`] but it is tied to a specific
/// path in the schema and with certain properties rendered for human readability.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayIndexEntry {
    /// Is this a spectrum or chromatogram array?
    pub context: BufferContext,
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
    /// The layout of buffer, either point or chunks
    pub buffer_format: BufferFormat,
    /// Any transformations applied to this array
    pub transform: Option<BufferTransform>,
    /// The default data processing method's ID applied to this array. Alternatives may
    /// be specified in the metadata table.
    pub data_processing_id: Option<Box<str>>,
    /// Whether this is the primary array for this quantity, or a secondary array.
    /// Primary arrays get a much more succinct standardized naming scheme while
    /// all other arrays' names are implementation details.
    pub buffer_priority: Option<BufferPriority>,
    /// In what rank order this array's values are sorted, if at all, within a parent
    /// entity. This may be useful when deciding how to prioritize applying filters.
    ///
    /// Lower values are sorted first.
    pub sorting_rank: Option<u32>,
}

impl ArrayIndexEntry {
    pub fn new(
        context: BufferContext,
        path: String,
        name: String,
        data_type: DataType,
        array_type: ArrayType,
        unit: Unit,
        buffer_format: BufferFormat,
    ) -> Self {
        Self {
            context,
            path,
            name,
            data_type,
            array_type,
            unit,
            buffer_format,
            transform: None,
            data_processing_id: None,
            buffer_priority: None,
            sorting_rank: None,
        }
    }

    /// Create an [`ArrayIndexEntry`] from a [`BufferName`] given some context.
    ///
    /// # Arguments
    /// - `prefix`: The parent path this array lives under
    /// - `bufer_name`: The description of the array to create an entry for
    /// - `field`: An extant column definition from a schema to get the column's name from instead of `buffer_name`, if available
    pub fn from_buffer_name(
        prefix: String,
        buffer_name: BufferName,
        field: Option<&Field>,
    ) -> Self {
        let path = [
            prefix.clone(),
            field
                .map(|f| f.name().to_string())
                .unwrap_or_else(|| buffer_name.to_string()),
        ]
        .join(".");

        Self {
            context: buffer_name.context,
            path,
            data_type: array_to_arrow_type(buffer_name.dtype),
            name: buffer_name.array_name(),
            array_type: buffer_name.array_type,
            unit: buffer_name.unit,
            buffer_format: buffer_name.buffer_format,
            transform: buffer_name.transform,
            data_processing_id: buffer_name.data_processing_id,
            buffer_priority: buffer_name.buffer_priority,
            sorting_rank: buffer_name.sorting_rank,
        }
    }

    /// Create a [`BufferName`] from this [`ArrayIndexEntry`]
    pub fn as_buffer_name(&self) -> BufferName {
        let mut this = BufferName::new_with_buffer_format(
            self.context,
            self.array_type.clone(),
            arrow_to_array_type(&self.data_type).unwrap(),
            self.buffer_format,
        )
        .with_transform(self.transform)
        .with_unit(self.unit)
        .with_priority(self.buffer_priority)
        .with_sorting_rank(self.sorting_rank);
        this.data_processing_id = self.data_processing_id.clone();
        this
    }

    /// Whether this describes an ion mobility array
    pub const fn is_ion_mobility(&self) -> bool {
        self.array_type.is_ion_mobility()
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
    entries: Vec<ArrayIndexEntry>,
}

impl ArrayIndex {
    pub fn new(prefix: String, entries: HashMap<ArrayType, ArrayIndexEntry>) -> Self {
        Self {
            prefix,
            entries: entries.into_values().collect(),
        }
    }

    pub fn get(&self, key: &ArrayType) -> Option<&ArrayIndexEntry> {
        self.entries
            .iter()
            .filter(|v| matches!(v.buffer_format, BufferFormat::Chunked | BufferFormat::Point))
            .find(|v| v.array_type == *key)
    }

    pub fn get_all(&self, key: &ArrayType) -> impl Iterator<Item = &ArrayIndexEntry> {
        self.entries
            .iter()
            .filter(|v| matches!(v.buffer_format, BufferFormat::Chunked | BufferFormat::Point))
            .filter(|v| v.array_type == *key)
    }

    pub fn as_slice(&self) -> &[ArrayIndexEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, k: &ArrayType) -> bool {
        self.get(k).is_some()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ArrayIndexEntry> {
        self.entries.iter()
    }

    pub fn push(&mut self, v: ArrayIndexEntry) {
        self.entries.push(v);
    }

    pub fn has_ion_mobility(&self) -> bool {
        self.entries.iter().any(|v| v.is_ion_mobility())
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
        let mut entries = Vec::new();
        for v in value.entries.into_iter() {
            let v = ArrayIndexEntry::from(v);
            entries.push(v);
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
            .into_iter()
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

/// A mapping from one [`BufferName`] to another [`BufferName`] that is used to
/// tell an [`ArrayBufferWriter`](crate::writer::ArrayBufferWriter) how to recast
/// data types, units, and other metadata. Needed to enforce consistent array typing
/// unless special care is taken.
#[derive(Debug, Default, Clone)]
pub struct BufferOverrideTable(HashMap<BufferName, BufferName>);

impl BufferOverrideTable {
    /// Check if a [`BufferName`] is overridden
    pub fn contains_key(&self, k: &BufferName) -> bool {
        self.0.contains_key(k)
    }

    /// Get the overriding [`BufferName`], if any
    fn get(&self, k: &BufferName) -> Option<&BufferName> {
        self.0.get(k)
    }

    /// See [`HashMap::iter`]
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, BufferName, BufferName> {
        self.0.iter()
    }

    /// See [`HashMap::iter_mut`]
    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<'_, BufferName, BufferName> {
        self.0.iter_mut()
    }

    /// See [`HashMap::keys`]
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, BufferName, BufferName> {
        self.0.keys()
    }

    /// Given a list of [`BufferName`] whose [`BufferPriority`] has been set, make sure all
    pub fn propagate_priorities(&mut self, buffer_names: &[BufferName]) {
        // Make sure all overrides for the same destination array type are equal priority
        for (_, val) in self.iter_mut() {
            for alt in buffer_names {
                // {
                if alt.array_type == val.array_type
                    && alt.unit == val.unit
                    && alt.dtype == val.dtype
                {
                    let before = val.buffer_priority;
                    if alt.buffer_priority != before {
                        let name_before = format!("{val:?}");
                        val.buffer_priority = val.buffer_priority.max(alt.buffer_priority);
                        log::debug!(
                            "{name_before} priority before {before:?} to {:?}",
                            val.buffer_priority
                        );
                    }
                }
            }
        }

        // Ensure that the provided buffer names are self-mapping
        for v in buffer_names {
            self.insert(v.clone(), v.clone());
        }
    }

    /// Map one [`BufferName`] to its pair, if it exists, or itself if no pair exists.
    ///
    /// The returned [`BufferName`] will have the greater of the two's [`BufferName::buffer_priority`]
    /// and will inherit the sorting rank of `k` or else of the mapped [`BufferName`].
    pub fn map(&self, k: &BufferName) -> BufferName {
        let mut name = self.get(k).or(Some(k)).cloned().unwrap();
        name.buffer_priority = k.buffer_priority.max(name.buffer_priority);
        name.sorting_rank = k.sorting_rank.or(name.sorting_rank);
        name
    }

    /// See [`HashMap::insert`]
    pub fn insert(&mut self, k: BufferName, v: BufferName) -> Option<BufferName> {
        self.0.insert(k, v)
    }

    /// See [`HashMap::remove`]
    pub fn remove(&mut self, k: &BufferName) -> Option<BufferName> {
        self.0.remove(k)
    }
}

impl From<HashMap<BufferName, BufferName>> for BufferOverrideTable {
    fn from(value: HashMap<BufferName, BufferName>) -> Self {
        Self(value)
    }
}

impl FromIterator<(BufferName, BufferName)> for BufferOverrideTable {
    fn from_iter<T: IntoIterator<Item = (BufferName, BufferName)>>(iter: T) -> Self {
        HashMap::from_iter(iter).into()
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_priority_cmp() {
        assert_eq!(
            None.max(Some(BufferPriority::Primary)),
            Some(BufferPriority::Primary)
        );
        assert_eq!(
            Some(BufferPriority::Primary).max(None),
            Some(BufferPriority::Primary)
        );
        assert_eq!(
            Some(BufferPriority::Secondary).max(Some(BufferPriority::Primary)),
            Some(BufferPriority::Primary)
        );
    }

    #[test]
    fn test_buffer_naming() {
        let name = BufferName::new(
            BufferContext::Chromatogram,
            ArrayType::TimeArray,
            BinaryDataArrayType::Float32,
        )
        .with_unit(Unit::Minute)
        .to_string();
        assert_eq!(name, "time_f32_min");

        let array_types = [
            ArrayType::Unknown,
            ArrayType::MZArray,
            ArrayType::IntensityArray,
            ArrayType::ChargeArray,
            ArrayType::SignalToNoiseArray,
            ArrayType::TimeArray,
            ArrayType::WavelengthArray,
            ArrayType::IonMobilityArray,
            ArrayType::MeanIonMobilityArray,
            ArrayType::RawIonMobilityArray,
            ArrayType::DeconvolutedIonMobilityArray,
            ArrayType::NonStandardDataArray {
                name: "frobnication level".to_string().into(),
            },
            ArrayType::BaselineArray,
            ArrayType::ResolutionArray,
            ArrayType::RawInverseReducedIonMobilityArray,
            ArrayType::MeanInverseReducedIonMobilityArray,
            ArrayType::RawDriftTimeArray,
            ArrayType::MeanDriftTimeArray,
        ];

        let units = [
            Unit::Unknown,
            Unit::MZ,
            Unit::Mass,
            Unit::PartsPerMillion,
            Unit::Nanometer,
            Unit::Minute,
            Unit::Second,
            Unit::Millisecond,
            Unit::VoltSecondPerSquareCentimeter,
            Unit::DetectorCounts,
            Unit::PercentBasePeak,
            Unit::PercentBasePeakTimes100,
            Unit::AbsorbanceUnit,
            Unit::CountsPerSecond,
            Unit::Electronvolt,
            Unit::Volt,
            Unit::Celsius,
            Unit::Kelvin,
            Unit::Pascal,
            Unit::Psi,
            Unit::MicrolitersPerMinute,
            Unit::Percent,
            Unit::Dimensionless,
        ];

        let dtypes = [
            BinaryDataArrayType::Float32,
            BinaryDataArrayType::Float64,
            BinaryDataArrayType::Int32,
            BinaryDataArrayType::Int64,
            BinaryDataArrayType::ASCII,
        ];

        let mut instances = Vec::new();

        // Prove that all combinations work
        for ((a, u), dtype) in array_types.iter().cartesian_product(units.iter()).cartesian_product(dtypes.iter()) {
            let mut name = BufferName::new(
                BufferContext::Spectrum,
                a.clone(),
                *dtype,
            );
            assert_ne!(name.to_string(), "");
            let no_primary = name.clone();
            instances.push(name.clone());

            name = name.with_unit(*u);
            assert_ne!(name.to_string(), "");
            instances.push(name.clone());

            name = name.with_priority(Some(BufferPriority::Primary));
            assert_ne!(name.to_string(), "");
            let primary = name.clone();
            assert!(no_primary < primary, "{no_primary:?} > {primary:?}");
            instances.push(name.clone());
        }

        instances.sort();
    }
}
