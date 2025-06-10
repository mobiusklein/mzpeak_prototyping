use mzdata::prelude::*;

use serde::{Deserialize, Serialize};

use crate::{
    CURIE, MZPeaksSelectedIonEntry, MzPeaksPrecursorEntry, MzPeaksScanEntry, MzPeaksSpectrumEntry,
    Param, peak_series::ToMzPeaksDataSeries,
};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksChromatogramEntry {
    pub index: Option<u64>,
    pub id: String,
    pub polarity: i8,
    pub chromatogram_type: CURIE,
    pub number_of_data_points: Option<u64>,
    pub parameters: Vec<Param>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct MzPeaksEntry {
    pub spectrum: Option<Box<MzPeaksSpectrumEntry>>,
    pub scan: Option<Box<MzPeaksScanEntry>>,
    pub precursor: Option<Box<MzPeaksPrecursorEntry>>,
    pub selected_ion: Option<Box<MZPeaksSelectedIonEntry>>,
    pub chromatogram: Option<Box<MzPeaksChromatogramEntry>>,
}

impl MzPeaksEntry {
    pub fn from_spectrum<
        C: ToMzPeaksDataSeries + CentroidLike,
        D: ToMzPeaksDataSeries + DeconvolutedCentroidLike,
    >(
        spectrum: &impl SpectrumLike<C, D>,
        index: Option<u64>,
    ) -> Vec<Self> {
        let mut spec = MzPeaksSpectrumEntry::from_spectrum(spectrum);
        if let Some(index) = index {
            spec.index = Some(index);
        }
        let index = spec.index;

        let mut entries = vec![Self {
            spectrum: Some(spec.into()),
            ..Default::default()
        }];

        for (i, event) in spectrum.acquisition().iter().enumerate() {
            if i == 0 {
                entries[0].scan = Some(MzPeaksScanEntry::from_scan_event(index, event).into());
            } else {
                entries.push(Self {
                    scan: Some(MzPeaksScanEntry::from_scan_event(index, event).into()),
                    ..Default::default()
                });
            }
        }

        if let Some(precursor) = spectrum.precursor() {
            let prec = MzPeaksPrecursorEntry::from_precursor(
                precursor,
                Some(spectrum.index() as u64),
                index,
            );
            entries[0].precursor = Some(prec.into());
            for (i, ion) in precursor.ions.iter().enumerate() {
                let part = MZPeaksSelectedIonEntry::from_selected_ion(
                    ion,
                    Some(spectrum.index() as u64),
                    index,
                );
                if i == 0 {
                    entries[0].selected_ion = Some(part.into())
                } else {
                    entries.push(part.into());
                }
            }
        }
        entries
    }
}

impl From<MzPeaksSpectrumEntry> for MzPeaksEntry {
    fn from(value: MzPeaksSpectrumEntry) -> Self {
        Self {
            spectrum: Some(Box::new(value)),
            ..Default::default()
        }
    }
}

impl From<MzPeaksScanEntry> for MzPeaksEntry {
    fn from(value: MzPeaksScanEntry) -> Self {
        Self {
            scan: Some(Box::new(value)),
            ..Default::default()
        }
    }
}

impl From<MzPeaksPrecursorEntry> for MzPeaksEntry {
    fn from(value: MzPeaksPrecursorEntry) -> Self {
        let mut this = Self::default();
        this.precursor = Some(Box::new(value));
        this
    }
}

impl From<MZPeaksSelectedIonEntry> for MzPeaksEntry {
    fn from(value: MZPeaksSelectedIonEntry) -> Self {
        let mut this = Self::default();
        this.selected_ion = Some(Box::new(value));
        this
    }
}
