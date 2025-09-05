use std::{fmt::Display, str::FromStr};

use mzdata::params::{ControlledVocabulary, ParamDescribed, ParamLike, Unit};
use serde::{Deserialize, Serialize};

/// Numerical identifier for "Proteomics Standards Initiative Mass Spectrometry Ontology"
pub const MS_CV_ID: u8 = 1;
/// Numerical identifier for "Unit Ontology"
pub const UO_CV_ID: u8 = 2;

/// A list of ion mobility point measures for scans
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

/// A controlled vocabulary or user parameter similar to [`mzdata::params::Param`]
///
/// This can be inter-converted with [`mzdata::params::Param`], though it is not a
/// zero-copy operation.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Param {
    /// The name, if given
    pub name: Option<String>,
    /// The accession code for the term
    pub accession: Option<CURIE>,
    /// A wide variant value type
    pub value: ParamValueSplit,
    /// The accession code for the unit
    pub unit: Option<CURIE>,
}

/// A numerical encoding of a CURIE accession similar to [`mzdata::params::CURIE`]
#[derive(
    Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub struct CURIE {
    pub cv_id: u8,
    pub accession: u32,
}

impl CURIE {
    pub fn format_with_namespace(&self, cvs: &[ControlledVocabulary]) -> Option<String> {
        let cv = cvs.get(self.cv_id.saturating_sub(1) as usize)?;
        Some(cv.curie(self.accession).to_string())
    }

    pub fn parse_with_namespace(
        text: &str,
        cvs: &[ControlledVocabulary],
    ) -> Result<Self, mzdata::params::CURIEParsingError> {
        let inst = mzdata::params::CURIE::from_str(text)?;
        if let Some(i) = cvs.iter().position(|i| *i == inst.controlled_vocabulary) {
            Ok(Self::new(i as u8, inst.accession))
        } else {
            Err(mzdata::params::CURIEParsingError::UnknownControlledVocabulary(
                mzdata::params::ControlledVocabularyResolutionError::UnknownControlledVocabulary(
                    format!("{:?}", inst.controlled_vocabulary)
                )))
        }
    }
}

impl FromStr for CURIE {
    type Err = mzdata::params::CURIEParsingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.parse::<mzdata::params::CURIE>()?.into())
    }
}

impl Display for CURIE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let alt: mzdata::params::CURIE = (*self).into();
        alt.fmt(f)
    }
}

impl From<Param> for mzdata::Param {
    fn from(value: Param) -> Self {
        let mut builder = mzdata::Param::builder();
        if let Some(curie) = value.accession {
            builder = builder.curie(curie.into())
        }
        if let Some(name) = value.name {
            builder = builder.name(name)
        }
        if let Some(unit) = value.unit {
            builder = builder.unit(Unit::from_curie(&unit.into()))
        }
        builder.value(value.value).build()
    }
}

impl From<&Param> for mzdata::Param {
    fn from(value: &Param) -> Self {
        let mut builder = mzdata::Param::builder();
        if let Some(curie) = value.accession {
            builder = builder.curie(curie.into())
        }
        if let Some(name) = value.name.clone() {
            builder = builder.name(name)
        }
        if let Some(unit) = value.unit {
            builder = builder.unit(Unit::from_curie(&unit.into()))
        }
        builder.value(value.value.clone()).build()
    }
}

#[macro_export]
macro_rules! curie {
    (MS:$acc:literal) => {
        CURIE {
            cv_id: 1,
            accession: $acc,
        }
    };
    (UO:$acc:literal) => {
        CURIE {
            cv_id: 2,
            accession: $acc,
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

impl From<CURIE> for mzdata::params::CURIE {
    fn from(value: CURIE) -> Self {
        Self {
            controlled_vocabulary: match value.cv_id {
                1 => mzdata::params::ControlledVocabulary::MS,
                2 => mzdata::params::ControlledVocabulary::UO,
                _ => todo!(),
            },
            accession: value.accession,
        }
    }
}

impl CURIE {
    pub fn new(cv_id: u8, accession: u32) -> Self {
        Self { cv_id, accession }
    }
}

// Provide a way to JSON-serialize CURIEs as nullable string
pub(crate) fn opt_curie_serialize<S>(
    curie: &Option<CURIE>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match curie {
        Some(curie) => serializer.serialize_str(&mzdata::params::CURIE::from(*curie).to_string()),
        None => serializer.serialize_none(),
    }
}

// Provide a way to JSON-serialize CURIEs as string
pub(crate) fn curie_serialize<S>(curie: &CURIE, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&mzdata::params::CURIE::from(*curie).to_string())
}

// Provide a way to JSON-deserialize CURIEs from a nullable string
pub(crate) fn opt_curie_deserialize<'de, D>(deserializer: D) -> Result<Option<CURIE>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CURIEVisit {}
    impl<'de> serde::de::Visitor<'de> for CURIEVisit {
        type Value = Option<CURIE>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("CURIE string or null")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            match v.parse::<mzdata::params::CURIE>() {
                Ok(v) => Ok(Some(v.into())),
                Err(e) => Err(E::custom(e)),
            }
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }
    }

    deserializer.deserialize_any(CURIEVisit {})
}

// Provide a way to JSON-deserialize CURIEs from a string
pub(crate) fn curie_deserialize<'de, D>(deserializer: D) -> Result<CURIE, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CURIEVisit {}
    impl<'de> serde::de::Visitor<'de> for CURIEVisit {
        type Value = CURIE;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("CURIE string")
        }

        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            match v.parse::<mzdata::params::CURIE>() {
                Ok(v) => Ok(v.into()),
                Err(e) => Err(E::custom(e)),
            }
        }
    }

    deserializer.deserialize_str(CURIEVisit {})
}

/// A wide-variant of [`mzdata::params::Value`] that can hold any strongly typed
/// parameter value.
///
/// The space of the other type variants is still allocated, which makes this structure
/// inefficient.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct ParamValueSplit {
    pub integer: Option<i64>,
    pub float: Option<f64>,
    pub boolean: Option<bool>,
    pub string: Option<String>,
}

impl From<ParamValueSplit> for mzdata::params::Value {
    fn from(value: ParamValueSplit) -> Self {
        if let Some(val) = value.boolean {
            mzdata::params::Value::Boolean(val)
        } else if let Some(val) = value.float {
            mzdata::params::Value::Float(val)
        } else if let Some(val) = value.integer {
            mzdata::params::Value::Int(val)
        } else if let Some(val) = value.string {
            mzdata::params::Value::String(val)
        } else {
            mzdata::params::Value::Empty
        }
    }
}

impl From<mzdata::params::Value> for ParamValueSplit {
    fn from(value: mzdata::params::Value) -> Self {
        let mut this = Self::default();
        match value {
            mzdata::params::Value::String(val) => this.string = Some(val),
            mzdata::params::Value::Float(val) => this.float = Some(val),
            mzdata::params::Value::Int(val) => this.integer = Some(val),
            mzdata::params::Value::Buffer(_) => todo!(),
            mzdata::params::Value::Boolean(val) => this.boolean = Some(val),
            mzdata::params::Value::Empty => {}
        }
        this
    }
}

impl From<mzdata::Param> for Param {
    fn from(value: mzdata::Param) -> Self {
        let curie = value.curie().map(CURIE::from);
        let val = ParamValueSplit::from(value.value);
        Self {
            name: Some(value.name),
            accession: curie,
            value: val,
            unit: value.unit.to_curie().map(CURIE::from),
        }
    }
}

/// A [`serde_json`]-friendly version of [`Param`] that uses
/// [`serde_json::Value`] instead of [`ParamValueSplit`].
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetaParam {
    pub name: Option<String>,
    #[serde(
        serialize_with = "opt_curie_serialize",
        deserialize_with = "opt_curie_deserialize"
    )]
    pub accession: Option<CURIE>,
    pub value: serde_json::Value,
    #[serde(
        serialize_with = "opt_curie_serialize",
        deserialize_with = "opt_curie_deserialize"
    )]
    pub unit: Option<CURIE>,
}

impl From<MetaParam> for mzdata::Param {
    fn from(value: MetaParam) -> Self {
        let mut this = Self::default();
        this.name = value.name.unwrap_or_default();
        this.unit = value
            .unit
            .map(|acc| Unit::from_curie(&(acc.into())))
            .unwrap_or_default();
        if let Some(curie) = value.accession {
            this.accession = Some(curie.accession);
            this.controlled_vocabulary = match curie.cv_id {
                MS_CV_ID => Some(ControlledVocabulary::MS),
                UO_CV_ID => Some(ControlledVocabulary::UO),
                _ => None,
            }
        }
        this.value = match value.value {
            serde_json::Value::Null => mzdata::params::Value::Empty,
            serde_json::Value::Bool(v) => mzdata::params::Value::Boolean(v),
            serde_json::Value::Number(number) => {
                if number.is_f64() {
                    mzdata::params::Value::Float(number.as_f64().unwrap())
                } else if number.is_i64() {
                    mzdata::params::Value::Int(number.as_i64().unwrap())
                } else {
                    mzdata::params::Value::Int(number.as_u64().unwrap() as i64)
                }
            }
            serde_json::Value::String(v) => mzdata::params::Value::String(v),
            serde_json::Value::Array(_) => mzdata::params::Value::String(value.value.to_string()),
            serde_json::Value::Object(_) => mzdata::params::Value::String(value.value.to_string()),
        };
        this
    }
}

impl From<mzdata::Param> for MetaParam {
    fn from(value: mzdata::Param) -> Self {
        let curie = value.curie().map(CURIE::from);
        let val = match value.value() {
            mzdata::params::ValueRef::String(x) => serde_json::Value::String(x.to_string()),
            mzdata::params::ValueRef::Float(x) => {
                serde_json::Value::Number(serde_json::Number::from_f64(x).unwrap())
            }
            mzdata::params::ValueRef::Int(x) => {
                serde_json::Value::Number(serde_json::Number::from_i128(x as i128).unwrap())
            }
            mzdata::params::ValueRef::Buffer(_) => unimplemented!(),
            mzdata::params::ValueRef::Empty => serde_json::Value::Null,
            mzdata::params::ValueRef::Boolean(x) => serde_json::Value::Bool(x),
        };
        Self {
            name: Some(value.name),
            accession: curie,
            value: val,
            unit: value.unit.to_curie().map(CURIE::from),
        }
    }
}

impl From<ParamValueSplit> for serde_json::Value {
    fn from(value: ParamValueSplit) -> Self {
        if let Some(val) = value.boolean {
            Self::from(val)
        } else if let Some(val) = value.float {
            Self::from(val)
        } else if let Some(val) = value.integer {
            Self::Number(serde_json::Number::from_i128(val as i128).unwrap())
        } else if let Some(val) = value.string {
            Self::from(val)
        } else {
            Self::Null
        }
    }
}

impl From<serde_json::Value> for ParamValueSplit {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::default(),
            serde_json::Value::Bool(x) => Self {
                boolean: Some(x),
                ..Default::default()
            },
            serde_json::Value::Number(number) => {
                if let Some(val) = number.as_f64() {
                    Self {
                        float: Some(val),
                        ..Default::default()
                    }
                } else if let Some(val) = number.as_i64() {
                    Self {
                        integer: Some(val),
                        ..Default::default()
                    }
                } else {
                    todo!()
                }
            }
            serde_json::Value::String(x) => Self {
                string: Some(x),
                ..Default::default()
            },
            serde_json::Value::Array(_values) => todo!(),
            serde_json::Value::Object(_map) => todo!(),
        }
    }
}

impl From<MetaParam> for Param {
    fn from(value: MetaParam) -> Self {
        Self {
            name: value.name,
            accession: value.accession,
            value: value.value.into(),
            unit: value.unit,
        }
    }
}

impl From<Param> for MetaParam {
    fn from(value: Param) -> Self {
        Self {
            name: value.name,
            accession: value.accession,
            value: value.value.into(),
            unit: value.unit,
        }
    }
}

/// An adaptation of [`mzdata::meta::SourceFile`]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub id: String,
    pub location: String,
    pub name: String,
    pub parameters: Vec<MetaParam>,
}

impl From<&mzdata::meta::SourceFile> for SourceFile {
    fn from(value: &mzdata::meta::SourceFile) -> Self {
        let mut parameters: Vec<MetaParam> = value
            .params()
            .iter()
            .cloned()
            .map(MetaParam::from)
            .collect();
        if let Some(p) = value.file_format.as_ref() {
            parameters.push(p.clone().into())
        }
        if let Some(p) = value.id_format.as_ref() {
            parameters.push(p.clone().into())
        }
        Self {
            id: value.id.clone(),
            location: value.location.clone(),
            name: value.name.clone(),
            parameters,
        }
    }
}

impl From<SourceFile> for mzdata::meta::SourceFile {
    fn from(value: SourceFile) -> Self {
        let mut params = Vec::new();
        let mut id_format = None;
        let mut file_format = None;
        for param in value.parameters {
            if let Some(curie) = param.accession {
                if let Some(term) = mzdata::meta::NativeSpectrumIdentifierFormatTerm::from_accession(
                    curie.accession,
                ) {
                    id_format = Some(term.into());
                } else if let Some(term) =
                    mzdata::meta::MassSpectrometerFileFormatTerm::from_accession(curie.accession)
                {
                    file_format = Some(term.into());
                } else {
                    params.push(param.into());
                }
            } else {
                params.push(param.into());
            }
        }

        Self {
            name: value.name,
            location: value.location,
            id: value.id,
            file_format,
            id_format,
            params,
        }
    }
}

/// An adaptation of [`mzdata::meta::FileDescription`]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FileDescription {
    pub contents: Vec<MetaParam>,
    pub source_files: Vec<SourceFile>,
}

impl From<FileDescription> for mzdata::meta::FileDescription {
    fn from(value: FileDescription) -> Self {
        let params: Vec<mzdata::params::Param> =
            value.contents.into_iter().map(|p| p.into()).collect();
        let source_files = value.source_files.into_iter().map(|sf| sf.into()).collect();
        Self::new(params, source_files)
    }
}

impl From<&mzdata::meta::FileDescription> for FileDescription {
    fn from(value: &mzdata::meta::FileDescription) -> Self {
        let contents = value
            .contents
            .iter()
            .cloned()
            .map(MetaParam::from)
            .collect();
        let source_files = value.source_files.iter().map(SourceFile::from).collect();
        Self {
            contents,
            source_files,
        }
    }
}

/// An adaptation of [`mzdata::meta::Software`]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Software {
    /// A unique identifier for the software within processing metadata
    pub id: String,
    /// A string denoting a particular software version, but does no guarantee is given for its format
    pub version: String,
    /// Any associated vocabulary terms, including actual software name and type
    pub parameters: Vec<MetaParam>,
}

impl From<Software> for mzdata::meta::Software {
    fn from(value: Software) -> Self {
        Self::new(
            value.id,
            value.version,
            value.parameters.into_iter().map(|p| p.into()).collect(),
        )
    }
}

impl From<&mzdata::meta::Software> for Software {
    fn from(value: &mzdata::meta::Software) -> Self {
        Self {
            id: value.id.clone(),
            version: value.version.clone(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

/// An adaptation of [`mzdata::meta::ProcessingMethod`]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ProcessingMethod {
    pub order: i8,
    pub software_reference: String,
    pub parameters: Vec<MetaParam>,
}

impl From<ProcessingMethod> for mzdata::meta::ProcessingMethod {
    fn from(value: ProcessingMethod) -> Self {
        Self {
            order: value.order,
            software_reference: value.software_reference,
            params: value.parameters.into_iter().map(|p| p.into()).collect(),
        }
    }
}

impl From<&mzdata::meta::ProcessingMethod> for ProcessingMethod {
    fn from(value: &mzdata::meta::ProcessingMethod) -> Self {
        Self {
            order: value.order,
            software_reference: value.software_reference.clone(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

/// An adaptation of [`mzdata::meta::DataProcessing`]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DataProcessing {
    pub id: String,
    pub methods: Vec<ProcessingMethod>,
}

impl From<DataProcessing> for mzdata::meta::DataProcessing {
    fn from(value: DataProcessing) -> Self {
        Self {
            id: value.id,
            methods: value.methods.into_iter().map(|p| p.into()).collect(),
        }
    }
}

impl From<&mzdata::meta::DataProcessing> for DataProcessing {
    fn from(value: &mzdata::meta::DataProcessing) -> Self {
        Self {
            id: value.id.clone(),
            methods: value.methods.iter().map(|v| v.into()).collect(),
        }
    }
}

/// An adaptation of [`mzdata::meta::ComponentType`]
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComponentType {
    /// A mass analyzer
    Analyzer,
    /// A source for ions
    IonSource,
    /// An abundance measuring device
    Detector,
    #[default]
    Unknown,
}

/// An adaptation of [`mzdata::meta::Component`]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Component {
    /// The kind of component this describes
    pub component_type: ComponentType,
    /// The order in the sequence of components that the analytes interact with
    pub order: u8,
    pub parameters: Vec<MetaParam>,
}

impl From<Component> for mzdata::meta::Component {
    fn from(value: Component) -> Self {
        Self {
            component_type: match value.component_type {
                ComponentType::Analyzer => mzdata::meta::ComponentType::Analyzer,
                ComponentType::IonSource => mzdata::meta::ComponentType::IonSource,
                ComponentType::Detector => mzdata::meta::ComponentType::Detector,
                ComponentType::Unknown => mzdata::meta::ComponentType::Unknown,
            },
            order: value.order,
            params: value
                .parameters
                .into_iter()
                .map(mzdata::Param::from)
                .collect(),
        }
    }
}

impl From<&mzdata::meta::Component> for Component {
    fn from(value: &mzdata::meta::Component) -> Self {
        Self {
            component_type: match value.component_type {
                mzdata::meta::ComponentType::Analyzer => ComponentType::Analyzer,
                mzdata::meta::ComponentType::IonSource => ComponentType::IonSource,
                mzdata::meta::ComponentType::Detector => ComponentType::Detector,
                mzdata::meta::ComponentType::Unknown => ComponentType::Unknown,
            },
            order: value.order,
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

/// An adaptation of [`mzdata::meta::InstrumentConfiguration`]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentConfiguration {
    /// The set of components involved
    pub components: Vec<Component>,
    /// A set of parameters that describe the instrument such as the model name or serial number
    pub parameters: Vec<MetaParam>,
    /// A reference to the data acquisition software involved in processing this configuration
    pub software_reference: String,
    /// A unique identifier translated to an ordinal identifying this configuration
    pub id: u32,
}

impl From<InstrumentConfiguration> for mzdata::meta::InstrumentConfiguration {
    fn from(value: InstrumentConfiguration) -> Self {
        Self {
            components: value.components.into_iter().map(|v| v.into()).collect(),
            params: value.parameters.into_iter().map(|v| v.into()).collect(),
            software_reference: value.software_reference,
            id: value.id,
        }
    }
}

impl From<&mzdata::meta::InstrumentConfiguration> for InstrumentConfiguration {
    fn from(value: &mzdata::meta::InstrumentConfiguration) -> Self {
        Self {
            components: value.components.iter().map(Component::from).collect(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
            software_reference: value.software_reference.clone(),
            id: value.id,
        }
    }
}

/// An adaptation of [`mzdata::meta::Sample`]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub id: String,
    pub name: Option<String>,
    pub parameters: Vec<MetaParam>,
}

impl From<Sample> for mzdata::meta::Sample {
    fn from(value: Sample) -> Self {
        Self {
            params: value.parameters.into_iter().map(|v| v.into()).collect(),
            name: value.name,
            id: value.id,
        }
    }
}

impl From<&mzdata::meta::Sample> for Sample {
    fn from(value: &mzdata::meta::Sample) -> Self {
        Self {
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
            name: value.name.clone(),
            id: value.id.clone(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataColumn {
    pub name: String,
    pub path: Vec<String>,
    pub index: usize,
    #[serde(
        serialize_with = "opt_curie_serialize",
        deserialize_with = "opt_curie_deserialize"
    )]
    pub accession: Option<CURIE>,
}

impl MetadataColumn {
    pub fn new(name: String, path: Vec<String>, index: usize, accession: Option<CURIE>) -> Self {
        Self {
            name,
            path,
            index,
            accession,
        }
    }

    pub fn scope(&self) -> &str {
        if self.path.len() <= 2 {
            ""
        } else {
            self.path.get(0).unwrap()
        }
    }

    pub fn create(&self, value: mzdata::params::Value, unit: Unit) -> mzdata::Param {
        if let Some(curie) = self.accession {
            let curie: mzdata::params::CURIE = curie.into();
            mzdata::Param {
                name: self.name.clone(),
                value,
                accession: Some(curie.accession),
                controlled_vocabulary: Some(curie.controlled_vocabulary),
                unit,
            }
        } else {
            mzdata::Param {
                name: self.name.clone(),
                value,
                accession: None,
                controlled_vocabulary: None,
                unit,
            }
        }
    }
}
