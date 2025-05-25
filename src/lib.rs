use mzdata::prelude::*;
use mzdata::spectrum::ArrayType;
use mzdata::{self, spectrum::ScanEvent};

use serde::{Deserialize, Serialize};

pub mod index;

pub const MS_CV_ID: u8 = 1;
pub const UO_CV_ID: u8 = 2;

pub const ION_MOBILITY_SCAN_TERMS: [mzdata::params::CURIE; 4] = [
    // ion mobility drift time
    mzdata::curie!(MS:1002476),
    // inverse reduced ion mobility drift time
    mzdata::curie!(MS:1002815),
    // FAIMS compensation voltage
    mzdata::curie!(MS:1001581),
    // SELEXION compensation voltage
    mzdata::curie!(MS:1003371),
];

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct PQParam {
    pub name: String,
    pub curie: CURIE,
    pub value: PQParamValue,
    pub unit: Option<CURIE>,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct CURIE {
    pub cv_id: u8,
    pub accession: u32,
}

macro_rules! curie {
    (MS:$acc:literal) => {
        CURIE {
            cv_id: 1,
            accession: $acc
        }
    };
    (UO:$acc:literal) => {
        CURIE {
            cv_id: 2,
            accession: $acc
        }
    };
}

impl From<mzdata::params::CURIE> for CURIE {
    fn from(value: mzdata::params::CURIE) -> Self {
        let cv_id = match value.controlled_vocabulary {
            mzdata::params::ControlledVocabulary::MS => 1,
            mzdata::params::ControlledVocabulary::UO => 2,
            mzdata::params::ControlledVocabulary::EFO => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::OBI => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::HANCESTRO => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::BFO => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::NCIT => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::BTO => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::PRIDE => panic!("Unsupported CV in {value}"),
            mzdata::params::ControlledVocabulary::Unknown => panic!("Unsupported CV in {value}"),
        };

        Self {
            cv_id,
            accession: value.accession,
        }
    }
}

impl CURIE {
    pub fn new(cv_id: u8, accession: u32) -> Self {
        Self { cv_id, accession }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct PQParamValue {
    pub integer: Option<i64>,
    pub float: Option<f64>,
    pub boolean: Option<bool>,
    pub string: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksSpectrumEntry {
    pub index: u64,
    pub id: String,
    pub ms_level: u8,
    pub time: f32,
    pub polarity: i8,
    pub mz_signal_continuity: CURIE,
    pub spectrum_type: CURIE,

    pub lowest_observed_mz: Option<f64>,
    pub highest_observed_mz: Option<f64>,

    pub lowest_observed_wavelength: Option<f64>,
    pub highest_observed_wavelength: Option<f64>,

    pub lowest_observed_ion_mobility: Option<f64>,
    pub highest_observed_ion_mobility: Option<f64>,

    pub number_of_data_points: Option<u64>,

    pub base_peak_mz: Option<f64>,
    pub base_peak_intensity: Option<f32>,
    pub total_ion_current: Option<f32>,

    pub parameters: Vec<PQParam>,
}

impl MzPeaksSpectrumEntry {
    pub fn from_spectrum(spectrum: &impl SpectrumLike) -> Self {
        let summaries = spectrum.peaks().fetch_summaries();

        let n_pts = summaries.len();
        let base_peak_mz = if n_pts > 0 {
            Some(summaries.base_peak.mz)
        } else {
            None
        };
        let base_peak_intensity = if n_pts > 0 {
            Some(summaries.base_peak.intensity)
        } else {
            None
        };

        Self {
            id: spectrum.id().into(),
            index: spectrum.index() as u64,
            ms_level: spectrum.ms_level(),
            time: spectrum.start_time() as f32,
            spectrum_type: match spectrum.ms_level() {
                0 => panic!("Unsupported ms level"),
                1 => CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1000579,
                },
                _ => CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1000580,
                },
            },
            polarity: match spectrum.polarity() {
                mzdata::spectrum::ScanPolarity::Unknown => 0,
                mzdata::spectrum::ScanPolarity::Positive => 1,
                mzdata::spectrum::ScanPolarity::Negative => -1,
            },
            mz_signal_continuity: match spectrum.signal_continuity() {
                mzdata::spectrum::SignalContinuity::Unknown => curie!(MS:1000525),
                mzdata::spectrum::SignalContinuity::Centroid => curie!(MS:1000127),
                mzdata::spectrum::SignalContinuity::Profile => curie!(MS:1000128),
            },

            lowest_observed_mz: Some(summaries.mz_range.0),
            highest_observed_mz: Some(summaries.mz_range.1),

            number_of_data_points: Some(n_pts as u64),
            base_peak_mz,
            base_peak_intensity,
            total_ion_current: Some(summaries.tic),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksScanEntry {
    pub spectrum_index: u64,
    pub scan_start_time: Option<f32>,
    pub preset_scan_configuration: Option<u32>,
    pub filter_string: Option<String>,
    pub ion_injection_time: Option<f32>,
    pub ion_mobility_value: Option<f64>,
    pub ion_mobility_type: Option<CURIE>,
}

impl MzPeaksScanEntry {
    pub fn from_scan_event(spectrum_index: u64, event: &ScanEvent) -> Self {
        let ion_mobility = event.ion_mobility();
        let ion_mobility_type = match ion_mobility {
            Some(_) => {
                let mut imt = None;
                for t in ION_MOBILITY_SCAN_TERMS.iter() {
                    if let Some(_) = event.get_param_by_curie(t) {
                        imt = Some(CURIE {
                            cv_id: MS_CV_ID,
                            accession: t.accession,
                        });
                        break;
                    }
                }
                imt
            }
            None => None,
        };

        Self {
            spectrum_index: spectrum_index,
            ion_injection_time: Some(event.injection_time),
            scan_start_time: Some(event.start_time as f32),
            preset_scan_configuration: Some(event.instrument_configuration_id),
            filter_string: event.filter_string().map(|s| s.to_string()),
            ion_mobility_value: ion_mobility,
            ion_mobility_type,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IsolationWindow {
    pub target: Option<f32>,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub parameters: Vec<PQParam>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Activation {
    pub parameters: Vec<PQParam>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksPrecursorEntry {
    pub spectrum_index: u64,
    pub isolation_window: IsolationWindow,
    pub activation: Activation,
}

pub trait MzPeakDataSeries: Serialize + Default + Clone {
    fn spectrum_index(&self) -> u64;
    fn array_names(&self) -> Vec<ArrayType>;
    fn from_spectrum<T: SpectrumLike>(spectrum: &T, spectrum_index: u64) -> Vec<Self>;
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksMZPoint {
    pub spectrum_index: u64,
    pub mz: f64,
    pub intensity: f32,
}

impl MzPeakDataSeries for MzPeaksMZPoint {
    fn spectrum_index(&self) -> u64 {
        self.spectrum_index
    }

    fn array_names(&self) -> Vec<ArrayType> {
        vec![ArrayType::MZArray, ArrayType::IntensityArray]
    }

    fn from_spectrum<T: SpectrumLike>(spectrum: &T, spectrum_index: u64) -> Vec<Self> {
        spectrum.peaks().iter().map(|p| {
            Self {
                spectrum_index,
                mz: p.mz,
                intensity: p.intensity
            }
        }).collect()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksMZIMPoint {
    pub spectrum_index: u64,
    pub mz: f64,
    pub intensity: f32,
    pub im: f64,
}

impl MzPeakDataSeries for MzPeaksMZIMPoint {
    fn spectrum_index(&self) -> u64 {
        self.spectrum_index
    }

    fn array_names(&self) -> Vec<ArrayType> {
        vec![ArrayType::MZArray, ArrayType::IntensityArray, ArrayType::IonMobilityArray]
    }

    fn from_spectrum<T: SpectrumLike>(spectrum: &T, spectrum_index: u64) -> Vec<Self> {
        let arrays = spectrum.raw_arrays().unwrap();
        let (ims, _im_type) = arrays.ion_mobility().unwrap();
        let mzs = arrays.mzs().unwrap();
        let ints = arrays.intensities().unwrap();
        mzs.iter().copied().zip(ints.iter().copied()).zip(ims.iter().copied()).map(|((mz, intensity), im)| {
            Self {
                spectrum_index,
                mz,
                intensity,
                im
            }
        }).collect()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksEntry<T: MzPeakDataSeries + 'static = MzPeaksMZPoint> {
    pub spectrum: Option<Box<MzPeaksSpectrumEntry>>,
    pub scan: Option<Box<MzPeaksScanEntry>>,
    pub point: Option<T>,
}

impl<T: MzPeakDataSeries + 'static> MzPeaksEntry<T> {
    pub fn from_spectrum(spectrum: &impl SpectrumLike) -> Vec<Self> {
        let spec = MzPeaksSpectrumEntry::from_spectrum(spectrum);
        let spec_index = spec.index;

        let mut entries: Vec<Self> = vec![spec.into()];
        entries.extend(spectrum
            .acquisition()
            .iter()
            .map(|e| MzPeaksScanEntry::from_scan_event(spec_index, e).into())
        );

        let peaks = T::from_spectrum(spectrum, spec_index);

        entries.extend(peaks.into_iter().map(|p| {
            Self {
                point: Some(p),
                ..Default::default()
            }
        }));

        entries
    }
}

impl<T: MzPeakDataSeries + 'static> From<MzPeaksSpectrumEntry> for MzPeaksEntry<T> {
    fn from(value: MzPeaksSpectrumEntry) -> Self {
        Self {
            spectrum: Some(Box::new(value)),
            ..Default::default()
        }
    }
}

impl<T: MzPeakDataSeries + 'static> From<MzPeaksScanEntry> for MzPeaksEntry<T> {
    fn from(value: MzPeaksScanEntry) -> Self {
        Self {
            scan: Some(Box::new(value)),
            ..Default::default()
        }
    }
}


#[cfg(test)]
mod tests {
    use std::{fs, io};
    use itertools::Itertools;
    use parquet::{basic::ZstdLevel, file::properties::{EnabledStatistics, WriterProperties}};
    use std::sync::Arc;
    use arrow::datatypes::FieldRef;
    use serde_arrow::schema::SchemaLike;

    use super::*;

    #[test]
    fn convert() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let entries: Vec<MzPeaksEntry> = reader
            .iter()
            .map(|s| MzPeaksEntry::from_spectrum(&s))
            .flatten()
            .collect();
        let fields = Vec::<FieldRef>::from_type::<MzPeaksEntry>(Default::default()).unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(fields.clone()));

        let handle = fs::File::create("small.mzpeak")?;
        let props = WriterProperties::builder().set_compression(parquet::basic::Compression::ZSTD(ZstdLevel::try_new(9).unwrap())).build();
        let mut writer = parquet::arrow::ArrowWriter::try_new(handle, schema, Some(props))?;
        let batch = serde_arrow::to_record_batch(&fields, &entries).unwrap();
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    #[test]
    fn convert_im() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("../mzdata/converted/E_20221108_EvoSepOne_3rdGenAurora15cm_CC_40SPD_whisper_scLF1108_M10_S1-A1_1_3103.zlib.mzML")?;
        let entries_iter = reader
            .iter()
            .map(|s| MzPeaksEntry::<MzPeaksMZIMPoint>::from_spectrum(&s)).chunks(1000);

        let fields = Vec::<FieldRef>::from_type::<MzPeaksEntry<MzPeaksMZIMPoint>>(Default::default()).unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(fields.clone()));

        let handle = fs::File::create("E_20221108_EvoSepOne_3rdGenAurora15cm_CC_40SPD_whisper_scLF1108_M10_S1-A1_1_3103.mzpeak")?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        let mut writer = parquet::arrow::ArrowWriter::try_new(handle, schema, Some(props))?;

        for (i, entries) in entries_iter.into_iter().enumerate() {
            eprintln!("Writing batch {i}");
            let chunk = entries.flatten().collect_vec();
            let batch = serde_arrow::to_record_batch(&fields, &chunk).unwrap();
            writer.write(&batch)?;
        }

        writer.finish()?;
        Ok(())
    }
}
