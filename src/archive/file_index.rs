use std::{collections::HashMap, ops::Deref, str::FromStr};

use serde::{Serialize, Deserialize};
use serde_with::DeserializeFromStr;


#[derive(Debug, Serialize, DeserializeFromStr, Clone)]
pub enum DataKind {
    #[serde(rename="data arrays")]
    DataArray,
    #[serde(rename="peaks")]
    Peaks,
    #[serde(rename="metadata")]
    Metadata,
    #[serde(rename="proprietary")]
    Proprietary,
    #[serde(rename="other")]
    Other(String),
}

impl FromStr for DataKind {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().trim() {
            "data arrays" => Self::DataArray,
            "peaks" => Self::Peaks,
            "metadata" => Self::Metadata,
            "proprietary" => Self::Proprietary,
            "other" => Self::Other("other".into()),
            _ => Self::Other(s.to_string())
        })
    }
}

#[derive(Debug, Serialize, DeserializeFromStr, Clone)]
pub enum EntityType {
    #[serde(rename="spectrum")]
    Spectrum,
    #[serde(rename="chromatogram")]
    Chromatogram,
    #[serde(rename="other")]
    Other(String),
}

impl FromStr for EntityType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().trim() {
            "spectrum" => Self::Spectrum,
            "chromatogram" => Self::Chromatogram,
            "other" => Self::Other("other".into()),
            _ => {
                log::warn!("Found entity type {s}, treating as 'other'");
                Self::Other(s.to_string())
            },
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub name: String,
    pub entity_type: EntityType,
    pub data_kind: DataKind,
}

impl FileEntry {
    pub fn archive_type(&self) -> super::MzPeakArchiveType {
        match (&self.entity_type, &self.data_kind) {
            (EntityType::Spectrum, DataKind::DataArray) => super::MzPeakArchiveType::SpectrumDataArrays,
            (EntityType::Spectrum, DataKind::Metadata) => super::MzPeakArchiveType::SpectrumMetadata,
            (EntityType::Spectrum, DataKind::Peaks) => super::MzPeakArchiveType::SpectrumPeakDataArrays,
            (EntityType::Chromatogram, DataKind::DataArray) => super::MzPeakArchiveType::ChromatogramDataArrays,
            (EntityType::Chromatogram, DataKind::Metadata) => super::MzPeakArchiveType::ChromatogramMetadata,
            (EntityType::Other(_), _) => super::MzPeakArchiveType::Other,
            (_, _) => {
                log::warn!("Could not map {self:?} to an archive type");
                super::MzPeakArchiveType::Other
            },
        }
    }

    pub fn new(name: String, entity_type: EntityType, data_kind: DataKind) -> Self {
        Self { name, entity_type, data_kind }
    }
}

impl From<super::MzPeakArchiveType> for FileEntry {
    fn from(value: super::MzPeakArchiveType) -> Self {
        match value {
            super::MzPeakArchiveType::SpectrumMetadata => FileEntry::new(value.tag_file_suffix().into(), EntityType::Spectrum, DataKind::Metadata),
            super::MzPeakArchiveType::SpectrumDataArrays => FileEntry::new(value.tag_file_suffix().into(), EntityType::Spectrum, DataKind::DataArray),
            super::MzPeakArchiveType::SpectrumPeakDataArrays => FileEntry::new(value.tag_file_suffix().into(), EntityType::Spectrum, DataKind::Peaks),
            super::MzPeakArchiveType::ChromatogramMetadata => FileEntry::new(value.tag_file_suffix().into(), EntityType::Chromatogram, DataKind::Metadata),
            super::MzPeakArchiveType::ChromatogramDataArrays => FileEntry::new(value.tag_file_suffix().into(), EntityType::Chromatogram, DataKind::DataArray),
            super::MzPeakArchiveType::Other => FileEntry::new("".into(), "other".parse().unwrap(), DataKind::Other("other".into())),
            super::MzPeakArchiveType::Proprietary => FileEntry::new("".into(), EntityType::Other("".into()), DataKind::Proprietary),
        }
    }
}


#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FileIndex {
    pub files: Vec<FileEntry>,
    pub metadata: HashMap<String, serde_json::Value>
}

impl From<Vec<FileEntry>> for FileIndex {
    fn from(value: Vec<FileEntry>) -> Self {
        Self::new(value, Default::default())
    }
}

impl FileIndex {
    pub const fn index_file_name() -> &'static str {
        "mzpeak_index.json"
    }

    pub fn new(files: Vec<FileEntry>, metadata: HashMap<String, serde_json::Value>) -> Self {
        Self { files, metadata }
    }

    pub fn push(&mut self, entry: FileEntry) {
        self.files.push(entry);
    }
}

impl Deref for FileIndex {
    type Target = [FileEntry];

    fn deref(&self) -> &Self::Target {
        &self.files
    }
}