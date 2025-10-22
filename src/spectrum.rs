use mzdata::{
    params::Unit,
    spectrum::{
        ArrayType, BinaryDataArrayType, DataArray,
        bindata::{ArrayRetrievalError, BinaryCompressionType},
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    param::{CURIE, MetadataColumn, Param, curie},
};

macro_rules! metacol {
    ($name:literal, $path:expr, $index:literal, $accession:expr) => {
        MetadataColumn::new(
            $name.into(),
            $path.into_iter().map(|v| v.into()).collect(),
            $index,
            Some($accession),
        )
    };
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuxiliaryArray {
    pub data: Vec<u8>,
    pub name: Param,
    pub data_type: CURIE,
    pub compression: CURIE,
    pub unit: Option<CURIE>,
    pub parameters: Vec<Param>,
    pub data_processing_ref: Option<Box<str>>,
}

impl From<AuxiliaryArray> for DataArray {
    fn from(value: AuxiliaryArray) -> Self {
        value.into_data_array()
    }
}

impl TryFrom<DataArray> for AuxiliaryArray {
    type Error = ArrayRetrievalError;
    fn try_from(value: DataArray) -> Result<Self, Self::Error> {
        Self::from_data_array(&value)
    }
}

impl AuxiliaryArray {
    pub fn into_data_array(self) -> DataArray {
        let mut chosen_compression = BinaryCompressionType::NoCompression;
        for method in BinaryCompressionType::COMPRESSION_METHODS.iter() {
            if let Some(method_param) = method.as_param() {
                let acc: CURIE = method_param.curie().unwrap().into();
                if acc == self.compression {
                    chosen_compression = *method;
                    break;
                }
            }
        }

        let data = if matches!(chosen_compression, BinaryCompressionType::NoCompression) {
            chosen_compression = BinaryCompressionType::Decoded;
            self.data
        } else {
            base64_simd::STANDARD.encode_type::<Vec<u8>>(&self.data)
        };

        let dtype_acc: mzdata::params::CURIE = self.data_type.into();
        let dtype = BinaryDataArrayType::from_accession(dtype_acc).unwrap();

        let name: mzdata::Param = self.name.into();
        let name = if let Some(name_acc) = name.curie() {
            if let Some(name_t) = ArrayType::from_accession(name_acc) {
                if matches!(name_t, ArrayType::NonStandardDataArray { name: _ }) {
                    ArrayType::nonstandard(name.value.to_string())
                } else {
                    name_t
                }
            } else {
                ArrayType::nonstandard(name.value.to_string())
            }
        } else {
            ArrayType::nonstandard(name.value.to_string())
        };

        let mut result = DataArray::wrap(&name, dtype, data);
        result.compression = chosen_compression;
        if !self.parameters.is_empty() {
            result.params = Some(Box::new(
                self.parameters
                    .into_iter()
                    .map(mzdata::Param::from)
                    .collect(),
            ));
        }
        result
    }

    pub fn from_data_array(
        source: &DataArray,
    ) -> Result<Self, mzdata::spectrum::bindata::ArrayRetrievalError> {
        let mut source = source.clone();
        source.decode_and_store()?;
        let data = source.data;

        let compression = BinaryCompressionType::NoCompression
            .as_param()
            .unwrap()
            .curie()
            .unwrap()
            .into();

        let data_type = source
            .dtype
            .curie()
            .ok_or_else(|| mzdata::spectrum::bindata::ArrayRetrievalError::DataTypeSizeMismatch)?
            .into();

        let unit = source.unit.to_curie().map(|c| c.into());

        let name = source.name.clone().as_param(Some(source.unit)).into();
        let mut this = Self {
            name,
            data,
            data_type,
            compression,
            unit,
            parameters: Default::default(),
            data_processing_ref: None,
        };
        if let Some(params) = source.params {
            this.parameters
                .extend(params.iter().map(|p| Param::from(p.clone())));
        }
        Ok(this)
    }
}

pub struct SpectrumEntry {}

impl SpectrumEntry {
    pub fn metadata_columns() -> Vec<MetadataColumn> {
        vec![
            MetadataColumn::new(
                "ms level".into(),
                vec!["spectrum".into(), "ms_level".into()],
                2,
                Some(curie!(MS:1000511)),
            ),
            MetadataColumn::new(
                "scan polarity".into(),
                vec!["spectrum".into(), "polarity".into()],
                4,
                Some(curie!(MS:1000465)),
            ),
            MetadataColumn::new(
                "spectrum representation".into(),
                vec!["spectrum".into(), "mz_signal_continuity".into()],
                5,
                Some(curie!(MS:1000525)),
            ),
            MetadataColumn::new(
                "spectrum type".into(),
                vec!["spectrum".into(), "spectrum_type".into()],
                6,
                Some(curie!(MS:1000559)),
            ),
            MetadataColumn::new(
                "lowest observed m/z".into(),
                vec!["spectrum".into(), "lowest_observed_mz".into()],
                7,
                Some(curie!(MS:1000528)),
            )
            .with_unit(Unit::MZ),
            MetadataColumn::new(
                "highest observed m/z".into(),
                vec!["spectrum".into(), "highest_observed_mz".into()],
                8,
                Some(curie!(MS:1000527)),
            )
            .with_unit(Unit::MZ),
            MetadataColumn::new(
                "lowest observed wavelength".into(),
                vec!["spectrum".into(), "lowest_observed_wavelength".into()],
                9,
                Some(curie!(MS:1000619)),
            ),
            MetadataColumn::new(
                "highest observed wavelength".into(),
                vec!["spectrum".into(), "highest_observed_wavelength".into()],
                10,
                Some(curie!(MS:1000618)),
            ),
            MetadataColumn::new(
                "lowest observed ion mobility".into(),
                vec!["spectrum".into(), "lowest_observed_ion_mobility".into()],
                11,
                Some(curie!(MS:1003439)),
            ),
            MetadataColumn::new(
                "highest observed ion mobility".into(),
                vec!["spectrum".into(), "highest_observed_ion_mobility".into()],
                12,
                Some(curie!(MS:1003440)),
            ),
            metacol!(
                "number of data points",
                vec!["spectrum", "number_of_data_points"],
                13,
                curie!(MS:1003060)
            ),
            MetadataColumn::new(
                "base peak m/z".into(),
                vec!["spectrum".into(), "base_peak_mz".into()],
                14,
                Some(curie!(MS:1000504)),
            )
            .with_unit(Unit::MZ),
            MetadataColumn::new(
                "base peak intensity".into(),
                vec!["spectrum".into(), "base_peak_intensity".into()],
                15,
                Some(curie!(MS:1000505)),
            ),
            MetadataColumn::new(
                "total ion current".into(),
                vec!["spectrum".into(), "total_ion_current".into()],
                16,
                Some(curie!(MS:1000285)),
            ),
        ]
    }
}

pub struct ScanWindowEntry {}


pub struct ScanEntry {}

impl ScanEntry {
    pub fn metadata_columns() -> Vec<MetadataColumn> {
        vec![
            metacol!(
                "scan start time",
                ["scan", "scan_start_time"],
                1,
                curie!(MS:1000016)
            )
            .with_unit(Unit::Minute),
            metacol!(
                "preset scan configuration",
                ["scan", "preset_scan_configuration"],
                2,
                curie!(MS:1000616)
            ),
            metacol!(
                "filter string",
                ["scan", "filter_string"],
                3,
                curie!(MS:1000512)
            ),
            metacol!(
                "ion injection time",
                ["scan", "ion_injection_time"],
                4,
                curie!(MS:1000927)
            )
            .with_unit(Unit::Millisecond),
        ]
    }
}

pub struct SelectedIonEntry {}

impl SelectedIonEntry {
    pub fn metadata_columns() -> Vec<MetadataColumn> {
        vec![
            metacol!(
                "selected ion m/z",
                vec!["selected_ion", "selected_ion_mz"],
                2,
                curie!(MS:1000744)
            )
            .with_unit(Unit::MZ),
            metacol!(
                "charge state",
                vec!["selected_ion", "charge_state"],
                3,
                curie!(MS:1000041)
            ),
            metacol!(
                "peak intensity",
                vec!["selected_ion", "intensity"],
                4,
                curie!(MS:1000042)
            )
            .with_unit(Unit::DetectorCounts),
        ]
    }
}

pub struct ChromatogramEntry {}

impl ChromatogramEntry {
    pub fn metadata_columns() -> Vec<MetadataColumn> {
        vec![
            MetadataColumn::new(
                "scan polarity".into(),
                vec!["chromatogram".into(), "polarity".into()],
                2,
                Some(curie!(MS:1000465)),
            ),
            metacol!(
                "chromatogram type",
                vec!["chromatogram", "chromatogram_type"],
                3,
                curie!(MS:1000626)
            ),
            metacol!(
                "number of data points",
                vec!["chromatogram", "number_of_data_points"],
                4,
                curie!(MS:1003060)
            ),
        ]
    }
}
