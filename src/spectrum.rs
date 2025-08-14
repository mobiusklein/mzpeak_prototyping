use mzdata::{
    params::{ControlledVocabulary, Unit},
    prelude::*,
    spectrum::{DataArray, IsolationWindowState, ScanEvent, bindata::BinaryCompressionType},
};
use serde::{Deserialize, Serialize};

use crate::{
    curie,
    param::{CURIE, ION_MOBILITY_SCAN_TERMS, MS_CV_ID, Param},
};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct AuxiliaryArray {
    pub data: Vec<u8>,
    pub name: Param,
    pub data_type: CURIE,
    pub compression: CURIE,
    pub unit: Option<CURIE>,
    pub parameters: Vec<Param>,
}

impl AuxiliaryArray {
    pub fn from_data_array(
        source: &DataArray,
    ) -> Result<Self, mzdata::spectrum::bindata::ArrayRetrievalError> {
        let mut source = source.clone();
        if source.compression == BinaryCompressionType::Decoded {
            source.store_compressed(BinaryCompressionType::Zstd)?;
        }
        let data = source.data;
        let data_type = source
            .dtype
            .curie()
            .ok_or_else(|| mzdata::spectrum::bindata::ArrayRetrievalError::DataTypeSizeMismatch)?
            .into();
        let unit = source.unit.to_curie().map(|c| c.into());
        let compression = source
            .compression
            .as_param()
            .unwrap()
            .curie()
            .unwrap()
            .into();
        let name = source.name.clone().as_param(Some(source.unit)).into();
        let mut this = Self {
            name,
            data,
            data_type,
            compression,
            unit,
            parameters: Default::default(),
        };
        if let Some(params) = source.params {
            this.parameters
                .extend(params.iter().map(|p| Param::from(p.clone())));
        }
        Ok(this)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct SpectrumEntry {
    pub index: Option<u64>,
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

    pub parameters: Vec<Param>,

    pub data_processing_ref: Option<u32>,
    pub auxiliary_arrays: Vec<AuxiliaryArray>,
    pub median_delta: Option<Vec<f64>>,
}

impl SpectrumEntry {
    pub fn from_spectrum<C: CentroidLike, D: DeconvolutedCentroidLike>(
        spectrum: &impl SpectrumLike<C, D>,
    ) -> Self {
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
            index: Some(spectrum.index() as u64),
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
            parameters: spectrum.params().iter().cloned().map(Param::from).collect(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct ScanEntry {
    pub spectrum_index: Option<u64>,
    pub scan_start_time: Option<f32>,
    pub preset_scan_configuration: Option<u32>,
    pub filter_string: Option<String>,
    pub ion_injection_time: Option<f32>,
    pub ion_mobility_value: Option<f64>,
    pub ion_mobility_type: Option<CURIE>,
    pub instrument_configuration_ref: Option<u32>,
    pub parameters: Vec<Param>,
}

impl ScanEntry {
    pub fn from_scan_event(spectrum_index: Option<u64>, event: &ScanEvent) -> Self {
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
            spectrum_index,
            ion_injection_time: Some(event.injection_time),
            scan_start_time: Some(event.start_time as f32),
            preset_scan_configuration: Some(event.instrument_configuration_id),
            filter_string: event.filter_string().map(|s| s.to_string()),
            ion_mobility_value: ion_mobility,
            ion_mobility_type,
            parameters: event.params().iter().cloned().map(Param::from).collect(),
            instrument_configuration_ref: Some(event.instrument_configuration_id),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct IsolationWindow {
    pub target: Option<f32>,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub parameters: Vec<Param>,
}

impl From<&mzdata::spectrum::IsolationWindow> for IsolationWindow {
    fn from(value: &mzdata::spectrum::IsolationWindow) -> Self {
        IsolationWindow {
            target: Some(value.target),
            lower_bound: Some(value.lower_bound),
            upper_bound: Some(value.upper_bound),
            parameters: Vec::new(),
        }
    }
}

impl From<IsolationWindow> for mzdata::spectrum::IsolationWindow {
    fn from(value: IsolationWindow) -> Self {
        let mut this = mzdata::spectrum::IsolationWindow::default();
        this.flags = IsolationWindowState::Explicit;
        let mut i = 0;
        if let Some(x) = value.lower_bound {
            this.lower_bound = x;
            i += 1;
        }
        if let Some(x) = value.target {
            this.target = x;
            i += 1;
        }
        if let Some(x) = value.upper_bound {
            this.upper_bound = x;
        }

        if i == 3 {
            this.flags = IsolationWindowState::Complete;
        }
        this
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Activation {
    pub parameters: Vec<Param>,
}

impl From<&mzdata::spectrum::Activation> for Activation {
    fn from(value: &mzdata::spectrum::Activation) -> Self {
        let mut parameters = Vec::new();
        for method in value.methods() {
            let par: mzdata::Param = method.to_param().into();
            parameters.push(par.into());
        }

        let energy = mzdata::Param::builder()
            .name("collision energy")
            .curie(mzdata::curie!(MS:1000045))
            .value(value.energy)
            .unit(Unit::Electronvolt)
            .build();
        parameters.push(energy.into());
        parameters.extend(value.params().iter().cloned().map(Param::from));

        Self { parameters }
    }
}

impl From<Activation> for mzdata::spectrum::Activation {
    fn from(value: Activation) -> Self {
        let mut this = mzdata::spectrum::Activation::default();
        for param in value.parameters {
            if let Some(acc) = param.accession.as_ref() {
                if *acc == curie!(MS:100045) {
                    this.energy = param.value.float.unwrap() as f32;
                } else {
                    this.add_param(param.into());
                }
            } else {
                this.add_param(param.into());
            }
        }
        this
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct PrecursorEntry {
    pub spectrum_index: Option<u64>,
    pub precursor_index: Option<u64>,
    pub precursor_id: Option<String>,
    pub selected_ion_count: u32,
    pub isolation_window: IsolationWindow,
    pub activation: Activation,
}

impl PrecursorEntry {
    pub fn from_precursor(
        precursor: &mzdata::spectrum::Precursor,
        spectrum_index: Option<u64>,
        precursor_index: Option<u64>,
    ) -> Self {
        Self {
            spectrum_index,
            precursor_index,
            precursor_id: precursor.precursor_id.clone(),
            selected_ion_count: precursor.ions.len() as u32,
            isolation_window: (&precursor.isolation_window).into(),
            activation: (&precursor.activation).into(),
        }
    }

    pub fn to_mzdata(&self) -> mzdata::spectrum::Precursor {
        let mut prec = mzdata::spectrum::Precursor::default();
        prec.isolation_window = self.isolation_window.clone().into();
        prec.activation = self.activation.clone().into();
        prec.precursor_id = self.precursor_id.clone();
        prec
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct SelectedIonEntry {
    pub spectrum_index: Option<u64>,
    pub precursor_index: Option<u64>,
    pub selected_ion_mz: Option<f64>,
    pub charge_state: Option<i32>,
    pub intensity: Option<f32>,
    pub ion_mobility: Option<f64>,
    pub ion_mobility_type: Option<CURIE>,
    pub parameters: Vec<Param>,
}

impl SelectedIonEntry {
    pub fn from_selected_ion(
        selected_ion: &mzdata::spectrum::SelectedIon,
        spectrum_index: Option<u64>,
        precursor_index: Option<u64>,
    ) -> Self {
        let mut ion_mobility_type = None;
        if selected_ion.has_ion_mobility() {
            for c in ION_MOBILITY_SCAN_TERMS {
                if let Some(im_type) = selected_ion.get_param_by_curie(&c) {
                    ion_mobility_type = im_type.curie().map(CURIE::from);
                    break;
                }
            }
        }
        SelectedIonEntry {
            spectrum_index,
            precursor_index,
            selected_ion_mz: Some(selected_ion.mz),
            charge_state: selected_ion.charge,
            intensity: Some(selected_ion.intensity),
            ion_mobility: selected_ion.ion_mobility(),
            ion_mobility_type,
            parameters: selected_ion
                .params()
                .iter()
                .cloned()
                .map(Param::from)
                .collect(),
            ..Default::default()
        }
    }

    pub fn to_mzdata(&self) -> mzdata::spectrum::SelectedIon {
        let mut si = mzdata::spectrum::SelectedIon::default();
        si.charge = self.charge_state;
        si.intensity = self.intensity.unwrap_or_default();
        si.mz = self.selected_ion_mz.unwrap_or_default();

        for p in self.parameters.iter().cloned() {
            si.add_param(p.into());
        }

        if let Some(im) = self.ion_mobility {
            let im_type: mzdata::params::CURIE = self.ion_mobility_type.unwrap().into();
            let im_param = mzdata::params::Param::builder();
            let im_param = match im_type {
                mzdata::params::CURIE {
                    controlled_vocabulary: ControlledVocabulary::MS,
                    accession: 1002476,
                } => im_param
                    .curie(im_type)
                    .name("ion mobility drift time")
                    .value(im)
                    .unit(Unit::Millisecond),
                mzdata::params::CURIE {
                    controlled_vocabulary: ControlledVocabulary::MS,
                    accession: 1002815,
                } => im_param
                    .curie(im_type)
                    .name("inverse reduced ion mobility drift time")
                    .value(im)
                    .unit(Unit::VoltSecondPerSquareCentimeter),
                mzdata::params::CURIE {
                    controlled_vocabulary: ControlledVocabulary::MS,
                    accession: 1001581,
                } => im_param
                    .curie(im_type)
                    .name("FAIMS compensation voltage")
                    .value(im)
                    .unit(Unit::Volt),
                mzdata::params::CURIE {
                    controlled_vocabulary: ControlledVocabulary::MS,
                    accession: 1003371,
                } => im_param
                    .curie(im_type)
                    .name("SELEXION compensation voltage")
                    .value(im)
                    .unit(Unit::Volt),
                _ => todo!("Don't know how to deal with {im_type}"),
            }
            .build();
            si.add_param(im_param);
        }

        for p in self.parameters.iter().map(|p| p.into()) {
            si.add_param(p);
        }

        si
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct ChromatogramEntry {
    pub index: Option<u64>,
    pub id: String,
    pub polarity: i8,
    pub chromatogram_type: CURIE,
    pub number_of_data_points: Option<u64>,
    pub parameters: Vec<Param>,
}
