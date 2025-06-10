use std::{fmt::Display, str::FromStr};

use mzdata::params::{ParamDescribed, ParamLike, Unit};
use serde::{Deserialize, Serialize};

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
pub struct Param {
    pub name: Option<String>,
    pub accession: Option<CURIE>,
    pub value: ParamValueSplit,
    pub unit: Option<CURIE>,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct CURIE {
    pub cv_id: u8,
    pub accession: u32,
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

pub(crate) fn curie_serialize<S>(curie: &CURIE, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&mzdata::params::CURIE::from(*curie).to_string())
}

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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksSourceFile {
    pub id: String,
    pub location: String,
    pub name: String,
    pub parameters: Vec<MetaParam>,
}

impl From<&mzdata::meta::SourceFile> for MzPeaksSourceFile {
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksFileDescription {
    pub contents: Vec<MetaParam>,
    pub source_files: Vec<MzPeaksSourceFile>,
}

impl From<&mzdata::meta::FileDescription> for MzPeaksFileDescription {
    fn from(value: &mzdata::meta::FileDescription) -> Self {
        let contents = value
            .contents
            .iter()
            .cloned()
            .map(MetaParam::from)
            .collect();
        let source_files = value
            .source_files
            .iter()
            .map(MzPeaksSourceFile::from)
            .collect();
        Self {
            contents,
            source_files,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksSoftware {
    /// A unique identifier for the software within processing metadata
    pub id: String,
    /// A string denoting a particular software version, but does no guarantee is given for its format
    pub version: String,
    /// Any associated vocabulary terms, including actual software name and type
    pub parameters: Vec<MetaParam>,
}

impl From<&mzdata::meta::Software> for MzPeaksSoftware {
    fn from(value: &mzdata::meta::Software) -> Self {
        Self {
            id: value.id.clone(),
            version: value.version.clone(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksProcessingMethod {
    pub order: i8,
    pub software_reference: String,
    pub parameters: Vec<MetaParam>,
}

impl From<&mzdata::meta::ProcessingMethod> for MzPeaksProcessingMethod {
    fn from(value: &mzdata::meta::ProcessingMethod) -> Self {
        Self {
            order: value.order,
            software_reference: value.software_reference.clone(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MzPeaksDataProcessing {
    pub id: String,
    pub methods: Vec<MzPeaksProcessingMethod>,
}

impl From<&mzdata::meta::DataProcessing> for MzPeaksDataProcessing {
    fn from(value: &mzdata::meta::DataProcessing) -> Self {
        Self {
            id: value.id.clone(),
            methods: value.methods.iter().map(|v| v.into()).collect(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Component {
    /// A mass analyzer
    Analyzer,
    /// A source for ions
    IonSource,
    /// An abundance measuring device
    Detector,
    #[default]
    Unknown,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MzPeaksComponent {
    /// The kind of component this describes
    pub component_type: Component,
    /// The order in the sequence of components that the analytes interact with
    pub order: u8,
    pub parameters: Vec<MetaParam>,
}

impl From<&mzdata::meta::Component> for MzPeaksComponent {
    fn from(value: &mzdata::meta::Component) -> Self {
        Self {
            component_type: match value.component_type {
                mzdata::meta::ComponentType::Analyzer => Component::Analyzer,
                mzdata::meta::ComponentType::IonSource => Component::IonSource,
                mzdata::meta::ComponentType::Detector => Component::Detector,
                mzdata::meta::ComponentType::Unknown => Component::Unknown,
            },
            order: value.order,
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MzPeaksInstrumentConfiguration {
    /// The set of components involved
    pub components: Vec<MzPeaksComponent>,
    /// A set of parameters that describe the instrument such as the model name or serial number
    pub parameters: Vec<MetaParam>,
    /// A reference to the data acquisition software involved in processing this configuration
    pub software_reference: String,
    /// A unique identifier translated to an ordinal identifying this configuration
    pub id: u32,
}

impl From<&mzdata::meta::InstrumentConfiguration> for MzPeaksInstrumentConfiguration {
    fn from(value: &mzdata::meta::InstrumentConfiguration) -> Self {
        Self {
            components: value
                .components
                .iter()
                .map(MzPeaksComponent::from)
                .collect(),
            parameters: value.iter_params().cloned().map(MetaParam::from).collect(),
            software_reference: value.software_reference.clone(),
            id: value.id,
        }
    }
}
