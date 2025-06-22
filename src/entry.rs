use mzdata::prelude::*;

use serde::{Deserialize, Serialize};

use crate::{
    CURIE, SelectedIonEntry, PrecursorEntry, ScanEntry, SpectrumEntry,
    param::Param, peak_series::ToMzPeakDataSeries,
};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct ChromatogramEntry {
    pub index: Option<u64>,
    pub id: String,
    pub polarity: i8,
    pub chromatogram_type: CURIE,
    pub number_of_data_points: Option<u64>,
    pub parameters: Vec<Param>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Entry {
    pub spectrum: Option<Box<SpectrumEntry>>,
    pub scan: Option<Box<ScanEntry>>,
    pub precursor: Option<Box<PrecursorEntry>>,
    pub selected_ion: Option<Box<SelectedIonEntry>>,
    pub chromatogram: Option<Box<ChromatogramEntry>>,
}

impl Entry {
    pub fn from_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        spectrum: &impl SpectrumLike<C, D>,
        index: Option<u64>,
        precursor_index: Option<&mut u64>,
    ) -> Vec<Self> {
        let mut spec = SpectrumEntry::from_spectrum(spectrum);
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
                entries[0].scan = Some(ScanEntry::from_scan_event(index, event).into());
            } else {
                entries.push(Self {
                    scan: Some(ScanEntry::from_scan_event(index, event).into()),
                    ..Default::default()
                });
            }
        }

        if let Some(precursor) = spectrum.precursor() {
            let prec = PrecursorEntry::from_precursor(
                precursor,
                Some(spectrum.index() as u64),
                precursor_index.as_ref().map(|v| **v),

            );
            let prec_index = prec.precursor_index;
            if let Some(pi) = precursor_index {
                *pi += 1
            }
            entries[0].precursor = Some(prec.into());
            for (i, ion) in precursor.ions.iter().enumerate() {
                let part = SelectedIonEntry::from_selected_ion(
                    ion,
                    Some(spectrum.index() as u64),
                    prec_index,
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

impl From<SpectrumEntry> for Entry {
    fn from(value: SpectrumEntry) -> Self {
        Self {
            spectrum: Some(Box::new(value)),
            ..Default::default()
        }
    }
}

impl From<ScanEntry> for Entry {
    fn from(value: ScanEntry) -> Self {
        Self {
            scan: Some(Box::new(value)),
            ..Default::default()
        }
    }
}

impl From<PrecursorEntry> for Entry {
    fn from(value: PrecursorEntry) -> Self {
        let mut this = Self::default();
        this.precursor = Some(Box::new(value));
        this
    }
}

impl From<SelectedIonEntry> for Entry {
    fn from(value: SelectedIonEntry) -> Self {
        let mut this = Self::default();
        this.selected_ion = Some(Box::new(value));
        this
    }
}
