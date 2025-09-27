use std::{fmt::Debug, sync::Arc};

use arrow::{
    array::{
        ArrayBuilder, ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, Int8Builder,
        Int32Builder, Int64Builder, LargeListBuilder, LargeStringBuilder, NullBuilder, StructArray,
        UInt8Builder, UInt32Builder, UInt64Builder,
    },
    datatypes::{DataType, Field, FieldRef},
};
use mzdata::{
    params::Unit,
    prelude::*,
    spectrum::{ScanPolarity, SpectrumDescription},
};

use crate::spectrum::AuxiliaryArray;

pub trait VisitorBase: Debug {
    fn schema(&self) -> Vec<FieldRef>;

    fn append_null(&mut self);

    fn as_struct_type(&self) -> DataType {
        DataType::Struct(self.schema().into())
    }
}

pub trait StructVisitor<T>: VisitorBase {
    fn append_value(&mut self, item: &T) -> bool;

    fn append_option(&mut self, item: Option<&T>) -> bool {
        if let Some(item) = item {
            self.append_value(item)
        } else {
            self.append_null();
            false
        }
    }
}

pub trait StructVisitorBuilder<T>: StructVisitor<T> + ArrayBuilder {}

impl<T, U> StructVisitorBuilder<T> for U where U: StructVisitor<T> + ArrayBuilder {}

macro_rules! field {
    ($name:literal, $typeexpr:expr) => {
        Arc::new(Field::new($name, $typeexpr, true))
    };
    ($name:literal, $typeexpr:expr, $nullable:expr) => {
        Arc::new(Field::new($name, $typeexpr, $nullable))
    };
    ($name:ident, $typeexpr:expr) => {
        Arc::new(Field::new($name, $typeexpr, true))
    };
}

macro_rules! finish_it {
    ($builder:expr) => {
        Arc::new($builder.finish()) as ArrayRef
    };
}

macro_rules! finish_cloned {
    ($builder:expr) => {
        Arc::new($builder.finish_cloned()) as ArrayRef
    };
}

macro_rules! anyways {
    () => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    };
}

/// Inflect a controlled vocabulary term to a Parquet-compatible column name.
///
/// This involves `${cv}_${accession}_${formatted_name}` where formatted name
/// is the name of the term with all non alphanumeric characters are replaced
/// with '_'.
pub fn inflect_cv_term_to_column_name(curie: mzdata::params::CURIE, name: &str) -> String {
    let cv_part = curie.to_string().replace(":", "_");
    let mut buffer = String::with_capacity(name.len() + cv_part.len() + 1);
    buffer.push_str(&cv_part);
    buffer.push('_');
    for c in name.replace("m/z", "mz").chars() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            buffer.push(c);
        } else {
            buffer.push('_');
        }
    }
    buffer
}

pub struct CustomBuilderFromParameter {
    curie: mzdata::params::CURIE,
    value: Box<dyn ArrayBuilder>,
    field: FieldRef,
}

impl Debug for CustomBuilderFromParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomBuilderFromParameter").field("curie", &self.curie).field("value", &"...").field("field", &self.field).finish()
    }
}

impl CustomBuilderFromParameter {
    pub fn from_spec(curie: mzdata::params::CURIE, name: &str, dtype: DataType) -> Self {
        let name = inflect_cv_term_to_column_name(curie, name);
        let field = field!(name, dtype.clone());
        match dtype {
            DataType::Null => Self {
                curie,
                field,
                value: Box::new(NullBuilder::new()),
            },
            DataType::Boolean => Self {
                curie,
                field,
                value: Box::new(BooleanBuilder::new()),
            },
            DataType::Int64 => Self {
                curie,
                field,
                value: Box::new(Int64Builder::new()),
            },
            DataType::Float64 => Self {
                curie,
                field,
                value: Box::new(Float64Builder::new()),
            },
            DataType::LargeUtf8 => Self {
                curie,
                field,
                value: Box::new(LargeStringBuilder::new()),
            },
            _ => unimplemented!("{dtype:?} is not supported by CustomBuilderFromParameter"),
        }
    }
}

impl VisitorBase for CustomBuilderFromParameter {
    fn schema(&self) -> Vec<FieldRef> {
        vec![self.field.clone()]
    }

    fn append_null(&mut self) {
        match self.field.data_type() {
            DataType::Null => {
                self.value
                    .as_any_mut()
                    .downcast_mut::<NullBuilder>()
                    .unwrap()
                    .append_empty_value();
            }
            DataType::Boolean => {
                self.value
                    .as_any_mut()
                    .downcast_mut::<BooleanBuilder>()
                    .unwrap()
                    .append_null();
            }
            DataType::Int64 => {
                self.value
                    .as_any_mut()
                    .downcast_mut::<Int64Builder>()
                    .unwrap()
                    .append_null();
            }
            DataType::Float64 => {
                self.value
                    .as_any_mut()
                    .downcast_mut::<Float64Builder>()
                    .unwrap()
                    .append_null();
            }
            DataType::LargeUtf8 => {
                self.value
                    .as_any_mut()
                    .downcast_mut::<LargeStringBuilder>()
                    .unwrap()
                    .append_null();
            }
            _ => panic!("Unsupported value type {:?}", self.field.data_type()),
        }
    }
}

impl<T> StructVisitor<T> for CustomBuilderFromParameter
where
    T: ParamDescribed,
{
    fn append_value(&mut self, item: &T) -> bool {
        if let Some(val) = item.get_param_by_curie(&self.curie) {
            match self.field.data_type() {
                DataType::Null => {
                    self.value
                        .as_any_mut()
                        .downcast_mut::<NullBuilder>()
                        .unwrap()
                        .append_empty_value();
                }
                DataType::Boolean => {
                    self.value
                        .as_any_mut()
                        .downcast_mut::<BooleanBuilder>()
                        .unwrap()
                        .append_option(val.to_bool().ok());
                }
                DataType::Int64 => {
                    self.value
                        .as_any_mut()
                        .downcast_mut::<Int64Builder>()
                        .unwrap()
                        .append_option(val.to_i64().ok());
                }
                DataType::Float64 => {
                    self.value
                        .as_any_mut()
                        .downcast_mut::<Float64Builder>()
                        .unwrap()
                        .append_option(val.to_f64().ok());
                }
                DataType::LargeUtf8 => {
                    self.value
                        .as_any_mut()
                        .downcast_mut::<LargeStringBuilder>()
                        .unwrap()
                        .append_option(if val.is_empty() {
                            None
                        } else {
                            Some(val.value().to_string())
                        });
                }
                _ => panic!("Unsupported value type {:?}", self.field.data_type()),
            }
            true
        } else {
            self.append_null();
            false
        }
    }
}

impl ArrayBuilder for CustomBuilderFromParameter {
    anyways!();

    fn len(&self) -> usize {
        self.value.len()
    }

    fn finish(&mut self) -> ArrayRef {
        self.value.finish()
    }

    fn finish_cloned(&self) -> ArrayRef {
        self.value.finish_cloned()
    }
}

#[derive(Debug, Default)]
pub struct CURIEBuilder {
    cv_id: UInt8Builder,
    accession: UInt32Builder,
}

impl VisitorBase for CURIEBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![
            field!("cv_id", DataType::UInt8),
            field!("accession", DataType::UInt32),
        ]
    }

    fn append_null(&mut self) {
        self.cv_id.append_null();
        self.accession.append_null();
    }
}

impl StructVisitor<crate::CURIE> for CURIEBuilder {
    fn append_value(&mut self, item: &crate::CURIE) -> bool {
        self.cv_id.append_value(item.cv_id);
        self.accession.append_value(item.accession);
        true
    }
}

impl StructVisitor<mzdata::params::CURIE> for CURIEBuilder {
    fn append_value(&mut self, item: &mzdata::params::CURIE) -> bool {
        let item: crate::CURIE = (*item).into();
        self.append_value(&item)
    }
}

impl ArrayBuilder for CURIEBuilder {
    fn len(&self) -> usize {
        self.cv_id.len()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(StructArray::new(
            self.schema().into(),
            vec![
                Arc::new(self.cv_id.finish()),
                Arc::new(self.accession.finish()),
            ],
            None,
        ))
    }

    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(StructArray::new(
            self.schema().into(),
            vec![
                Arc::new(self.cv_id.finish_cloned()),
                Arc::new(self.accession.finish_cloned()),
            ],
            None,
        ))
    }

    anyways!();
}

#[derive(Debug, Default)]
pub struct ParamValueBuilder {
    integer: Int64Builder,
    float: Float64Builder,
    boolean: BooleanBuilder,
    string: LargeStringBuilder,
}

impl VisitorBase for ParamValueBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let fields = vec![
            Arc::new(Field::new("integer", DataType::Int64, true)),
            Arc::new(Field::new("float", DataType::Float64, true)),
            Arc::new(Field::new("string", DataType::LargeUtf8, true)),
            Arc::new(Field::new("boolean", DataType::Boolean, true)),
        ];
        fields
    }

    fn append_null(&mut self) {
        self.boolean.append_null();
        self.integer.append_null();
        self.string.append_null();
        self.float.append_null();
    }
}

impl StructVisitor<mzdata::params::Value> for ParamValueBuilder {
    fn append_value(&mut self, item: &mzdata::params::Value) -> bool {
        match item {
            mzdata::params::Value::String(v) => {
                self.string.append_value(v);
                self.integer.append_null();
                self.float.append_null();
                self.boolean.append_null();
                true
            }
            mzdata::params::Value::Float(v) => {
                self.string.append_null();
                self.integer.append_null();
                self.float.append_value(*v);
                self.boolean.append_null();
                true
            }
            mzdata::params::Value::Int(v) => {
                self.string.append_null();
                self.integer.append_value(*v);
                self.float.append_null();
                self.boolean.append_null();
                true
            }
            mzdata::params::Value::Buffer(_) => todo!(),
            mzdata::params::Value::Boolean(v) => {
                self.string.append_null();
                self.integer.append_null();
                self.float.append_null();
                self.boolean.append_value(*v);
                true
            }
            mzdata::params::Value::Empty => {
                self.string.append_null();
                self.integer.append_null();
                self.float.append_null();
                self.boolean.append_null();
                true
            }
        }
    }
}

impl StructVisitor<crate::param::ParamValueSplit> for ParamValueBuilder {
    fn append_value(&mut self, item: &crate::param::ParamValueSplit) -> bool {
        self.string.append_option(item.string.as_ref());
        self.integer.append_option(item.integer);
        self.float.append_option(item.float);
        self.boolean.append_option(item.boolean);
        true
    }
}

impl ArrayBuilder for ParamValueBuilder {
    fn len(&self) -> usize {
        self.string.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.integer.finish()),
            Arc::new(self.float.finish()),
            Arc::new(self.string.finish()),
            Arc::new(self.boolean.finish()),
        ];
        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.integer.finish_cloned()),
            Arc::new(self.float.finish_cloned()),
            Arc::new(self.boolean.finish_cloned()),
            Arc::new(self.string.finish_cloned()),
        ];
        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self as &mut dyn std::any::Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self as Box<dyn std::any::Any>
    }
}

#[derive(Debug, Default)]
pub struct ParamBuilder {
    value: ParamValueBuilder,
    curie: CURIEBuilder,
    name: LargeStringBuilder,
    unit: CURIEBuilder,
}

impl VisitorBase for ParamBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![
            field!("value", self.value.as_struct_type()),
            field!("curie", self.curie.as_struct_type()),
            field!("name", DataType::LargeUtf8),
            field!("unit", self.unit.as_struct_type()),
        ]
    }

    fn append_null(&mut self) {
        self.value.append_null();
        self.curie.append_null();
        self.name.append_null();
        self.unit.append_null();
    }
}

impl StructVisitor<crate::param::Param> for ParamBuilder {
    fn append_value(&mut self, item: &crate::param::Param) -> bool {
        self.name.append_option(item.name.as_ref());
        self.curie.append_option(item.accession.as_ref());
        self.unit.append_option(item.unit.as_ref());
        self.value.append_value(&item.value)
    }
}

impl StructVisitor<mzdata::Param> for ParamBuilder {
    fn append_value(&mut self, item: &mzdata::Param) -> bool {
        self.curie.append_option(item.curie().as_ref());
        self.name.append_value(item.name());
        self.unit.append_option(item.unit.to_curie().as_ref());
        self.value.append_value(&item.value)
    }
}

impl ArrayBuilder for ParamBuilder {
    fn len(&self) -> usize {
        self.name.len()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(StructArray::new(
            self.schema().into(),
            vec![
                self.value.finish(),
                self.curie.finish(),
                Arc::new(self.name.finish()),
                self.unit.finish(),
            ],
            None,
        ))
    }

    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(StructArray::new(
            self.schema().into(),
            vec![
                self.value.finish_cloned(),
                self.curie.finish_cloned(),
                Arc::new(self.name.finish_cloned()),
                self.unit.finish_cloned(),
            ],
            None,
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self as &mut dyn std::any::Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self as Box<dyn std::any::Any>
    }
}

#[derive(Debug, Default)]
pub struct ParamListBuilder(LargeListBuilder<ParamBuilder>);

impl ParamListBuilder {
    pub fn append_empty(&mut self) {
        self.0.append(true);
    }

    pub fn as_mut(&mut self) -> &mut LargeListBuilder<ParamBuilder> {
        &mut self.0
    }
}

impl ArrayBuilder for ParamListBuilder {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }

    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(self.0.finish_cloned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self as &mut dyn std::any::Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self as Box<dyn std::any::Any>
    }
}

impl<T> StructVisitor<&[T]> for ParamListBuilder
where
    ParamBuilder: StructVisitor<T> + Sized,
{
    fn append_value(&mut self, item: &&[T]) -> bool {
        let inner = self.0.values();
        for v in item.into_iter() {
            inner.append_value(v);
        }
        self.0.append(true);
        true
    }
}

impl VisitorBase for ParamListBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![field!(
            "parameters",
            DataType::LargeList(field!("item", self.0.values_ref().as_struct_type()))
        )]
    }

    fn append_null(&mut self) {
        self.0.append_null();
    }
}

#[derive(Debug, Default)]
pub struct ScanWindowBuilder {
    lower_limit: Float32Builder,
    upper_limit: Float32Builder,
    unit: CURIEBuilder,
    parameters: ParamListBuilder,
}

impl VisitorBase for ScanWindowBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = vec![
            field!("lower_limit", DataType::Float32),
            field!("upper_limit", DataType::Float32),
            field!("unit", self.unit.as_struct_type()),
        ];
        fields.extend(self.parameters.schema());
        fields
    }

    fn append_null(&mut self) {
        self.lower_limit.append_null();
        self.upper_limit.append_null();
        self.unit.append_null();
        self.parameters.append_null();
    }
}

impl StructVisitor<mzdata::spectrum::ScanWindow> for ScanWindowBuilder {
    fn append_value(&mut self, item: &mzdata::spectrum::ScanWindow) -> bool {
        self.lower_limit.append_value(item.lower_bound);
        self.upper_limit.append_value(item.upper_bound);
        self.unit.append_option(Unit::MZ.to_curie().as_ref());
        self.parameters.append_empty();
        true
    }
}

impl ArrayBuilder for ScanWindowBuilder {
    fn len(&self) -> usize {
        self.lower_limit.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();
        let arrays = vec![
            finish_it!(self.lower_limit),
            finish_it!(self.upper_limit),
            self.unit.finish(),
            self.parameters.finish(),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();
        let arrays = vec![
            Arc::new(self.lower_limit.finish_cloned()),
            Arc::new(self.upper_limit.finish_cloned()),
            self.unit.finish_cloned(),
            self.parameters.finish_cloned(),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    anyways!();
}

#[derive(Default, Debug)]
pub struct ScanBuilder {
    spectrum_index: UInt64Builder,
    scan_start_time: Float32Builder,
    preset_scan_configuration: UInt32Builder,
    filter_string: LargeStringBuilder,
    ion_injection_time: Float32Builder,
    ion_mobility_value: Float64Builder,
    ion_mobility_type: CURIEBuilder,
    instrument_configuration_ref: UInt32Builder,
    parameters: ParamListBuilder,
    scan_windows: LargeListBuilder<ScanWindowBuilder>,
    extra: Vec<Box<dyn StructVisitorBuilder<mzdata::spectrum::ScanEvent>>>,
}

impl VisitorBase for ScanBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = vec![
            field!("spectrum_index", DataType::UInt64),
            field!("scan_start_time", DataType::Float32),
            field!("preset_scan_configuration", DataType::UInt32),
            field!("filter_string", DataType::LargeUtf8),
            field!("ion_injection_time", DataType::Float32),
            field!("ion_mobility_value", DataType::Float64),
            field!("ion_mobility_type", self.ion_mobility_type.as_struct_type()),
            field!("instrument_configuration_ref", DataType::UInt32),
        ];
        fields.extend(self.parameters.schema());
        fields.push(field!(
            "scan_windows",
            DataType::LargeList(field!(
                "item",
                self.scan_windows.values_ref().as_struct_type()
            ))
        ));
        for e in self.extra.iter() {
            fields.extend(e.schema());
        }
        fields
    }

    fn append_null(&mut self) {
        self.spectrum_index.append_null();
        self.scan_start_time.append_null();
        self.preset_scan_configuration.append_null();
        self.filter_string.append_null();
        self.ion_injection_time.append_null();
        self.ion_mobility_value.append_null();
        self.ion_mobility_type.append_null();
        self.instrument_configuration_ref.append_null();
        self.parameters.append_null();
        self.scan_windows.append_null();
        for e in self.extra.iter_mut() {
            e.append_null();
        }
    }
}

impl StructVisitor<(u64, &mzdata::spectrum::ScanEvent)> for ScanBuilder {
    fn append_value(&mut self, item: &(u64, &mzdata::spectrum::ScanEvent)) -> bool {
        let (si, item) = item;
        self.spectrum_index.append_value(*si);
        self.scan_start_time.append_value(item.start_time as f32);
        self.preset_scan_configuration.append_option(
            item.scan_configuration()
                .map(|i| i.to_u64().unwrap() as u32),
        );
        self.filter_string
            .append_option(item.filter_string().as_deref());
        self.ion_injection_time.append_value(item.injection_time);
        self.ion_mobility_value.append_option(item.ion_mobility());
        self.ion_mobility_type
            .append_option(item.ion_mobility_type().and_then(|v| v.curie()).as_ref());
        self.instrument_configuration_ref
            .append_value(item.instrument_configuration_id);
        self.parameters.append_value(&item.params());

        let val = self.scan_windows.values();
        for window in item.scan_windows.iter() {
            val.append_value(window);
        }
        self.scan_windows.append(true);

        for e in self.extra.iter_mut() {
            e.append_value(item);
        }
        true
    }
}

impl ArrayBuilder for ScanBuilder {
    fn len(&self) -> usize {
        self.spectrum_index.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let schema = self.schema();
        let mut arrays: Vec<ArrayRef> = vec![
            finish_it!(self.spectrum_index),
            finish_it!(self.scan_start_time),
            finish_it!(self.preset_scan_configuration),
            finish_it!(self.filter_string),
            finish_it!(self.ion_injection_time),
            finish_it!(self.ion_mobility_value),
            self.ion_mobility_type.finish(),
            finish_it!(self.instrument_configuration_ref),
            self.parameters.finish(),
            finish_it!(self.scan_windows),
        ];
        for e in self.extra.iter_mut() {
            arrays.push(e.finish());
        }
        Arc::new(StructArray::new(schema.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let schema = self.schema();
        let mut arrays: Vec<ArrayRef> = vec![
            finish_cloned!(self.spectrum_index),
            finish_cloned!(self.scan_start_time),
            finish_cloned!(self.preset_scan_configuration),
            finish_cloned!(self.filter_string),
            finish_cloned!(self.ion_injection_time),
            finish_cloned!(self.ion_mobility_value),
            self.ion_mobility_type.finish_cloned(),
            finish_cloned!(self.instrument_configuration_ref),
            self.parameters.finish_cloned(),
        ];
        for e in self.extra.iter() {
            arrays.push(e.finish_cloned());
        }
        Arc::new(StructArray::new(schema.into(), arrays, None))
    }

    anyways!();
}

#[derive(Default, Debug)]
pub struct IsolationWindowBuilder {
    target: Float32Builder,
    lower_bound: Float32Builder,
    upper_bound: Float32Builder,
    parameters: ParamListBuilder,
}

impl ArrayBuilder for IsolationWindowBuilder {
    anyways!();

    fn len(&self) -> usize {
        self.target.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let schema = self.schema();
        let arrays: Vec<ArrayRef> = vec![
            finish_it!(self.target),
            finish_it!(self.lower_bound),
            finish_it!(self.upper_bound),
            self.parameters.finish(),
        ];

        Arc::new(StructArray::new(schema.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let schema = self.schema();
        let arrays: Vec<ArrayRef> = vec![
            finish_cloned!(self.target),
            finish_cloned!(self.lower_bound),
            finish_cloned!(self.upper_bound),
            self.parameters.finish_cloned(),
        ];

        Arc::new(StructArray::new(schema.into(), arrays, None))
    }
}

impl StructVisitor<mzdata::spectrum::IsolationWindow> for IsolationWindowBuilder {
    fn append_value(&mut self, item: &mzdata::spectrum::IsolationWindow) -> bool {
        self.target.append_value(item.target);
        self.lower_bound.append_value(item.lower_bound);
        self.upper_bound.append_value(item.upper_bound);
        self.parameters.append_empty();
        true
    }
}

impl VisitorBase for IsolationWindowBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = vec![
            field!("target", DataType::Float32),
            field!("lower_bound", DataType::Float32),
            field!("upper_bound", DataType::Float32),
        ];
        fields.extend(self.parameters.schema());
        fields
    }

    fn append_null(&mut self) {
        self.target.append_null();
        self.lower_bound.append_null();
        self.upper_bound.append_null();
        self.parameters.append_null();
    }
}

#[derive(Default, Debug)]
pub struct ActivationBuilder {
    parameters: ParamListBuilder,
    extra: Vec<Box<dyn StructVisitorBuilder<mzdata::spectrum::Activation>>>,
}

impl ArrayBuilder for ActivationBuilder {
    anyways!();

    fn len(&self) -> usize {
        self.parameters.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();
        let mut arrays = vec![self.parameters.finish()];

        for e in self.extra.iter_mut() {
            arrays.push(e.finish());
        }
        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();
        let mut arrays = vec![self.parameters.finish_cloned()];

        for e in self.extra.iter() {
            arrays.push(e.finish_cloned());
        }
        Arc::new(StructArray::new(fields.into(), arrays, None))
    }
}

impl StructVisitor<mzdata::spectrum::Activation> for ActivationBuilder {
    fn append_value(&mut self, item: &mzdata::spectrum::Activation) -> bool {
        let params = self.parameters.as_mut().values();
        for method in item.methods() {
            let par: mzdata::Param = method.to_param().into();
            params.append_value(&par);
        }

        let energy = mzdata::Param::builder()
            .name("collision energy")
            .curie(mzdata::curie!(MS:1000045))
            .value(item.energy)
            .unit(Unit::Electronvolt)
            .build();
        params.append_value(&energy);

        for p in item.params() {
            params.append_value(p);
        }

        self.parameters.as_mut().append(true);
        for e in self.extra.iter_mut() {
            e.append_value(item);
        }
        true
    }
}

impl VisitorBase for ActivationBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = self.parameters.schema();
        for e in self.extra.iter() {
            fields.extend(e.schema());
        }
        fields
    }

    fn append_null(&mut self) {
        self.parameters.append_null();
        self.extra.iter_mut().for_each(|e| e.append_null());
    }
}

#[derive(Default, Debug)]
pub struct PrecursorBuilder {
    spectrum_index: UInt64Builder,
    precursor_index: UInt64Builder,
    isolation_window: IsolationWindowBuilder,
    activation: ActivationBuilder,
}

impl ArrayBuilder for PrecursorBuilder {
    fn len(&self) -> usize {
        self.spectrum_index.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();

        let arrays = vec![
            finish_it!(self.spectrum_index),
            finish_it!(self.precursor_index),
            self.isolation_window.finish(),
            self.activation.finish(),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();

        let arrays = vec![
            finish_cloned!(self.spectrum_index),
            finish_cloned!(self.precursor_index),
            self.isolation_window.finish_cloned(),
            self.activation.finish_cloned(),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    anyways!();
}

impl StructVisitor<(u64, u64, &mzdata::spectrum::Precursor)> for PrecursorBuilder {
    fn append_value(&mut self, item: &(u64, u64, &mzdata::spectrum::Precursor)) -> bool {
        let (i, j, item) = item;
        self.spectrum_index.append_value(*i);
        self.precursor_index.append_value(*j);

        self.isolation_window.append_value(&item.isolation_window);
        self.activation.append_value(&item.activation);
        true
    }
}

impl VisitorBase for PrecursorBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![
            field!("spectrum_index", DataType::UInt64),
            field!("precursor_index", DataType::UInt64),
            field!("isolation_window", self.isolation_window.as_struct_type()),
            field!("activation", self.activation.as_struct_type()),
        ]
    }

    fn append_null(&mut self) {
        self.spectrum_index.append_null();
        self.precursor_index.append_null();
        self.isolation_window.append_null();
        self.activation.append_null();
    }
}

#[derive(Default, Debug)]
pub struct SelectedIonBuilder {
    spectrum_index: UInt64Builder,
    precursor_index: UInt64Builder,
    selected_ion_mz: Float64Builder,
    charge_state: Int32Builder,
    intensity: Float32Builder,
    ion_mobility: Float64Builder,
    ion_mobility_type: CURIEBuilder,
    parameters: ParamListBuilder,
    extra: Vec<Box<dyn StructVisitorBuilder<mzdata::spectrum::SelectedIon>>>,
}

impl ArrayBuilder for SelectedIonBuilder {
    fn len(&self) -> usize {
        self.spectrum_index.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();

        let mut arrays = vec![
            finish_it!(self.spectrum_index),
            finish_it!(self.precursor_index),
            finish_it!(self.selected_ion_mz),
            finish_it!(self.charge_state),
            finish_it!(self.intensity),
            finish_it!(self.ion_mobility),
            self.ion_mobility_type.finish(),
            self.parameters.finish(),
        ];

        for e in self.extra.iter_mut() {
            arrays.push(e.finish());
        }

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();

        let mut arrays = vec![
            finish_cloned!(self.spectrum_index),
            finish_cloned!(self.precursor_index),
            finish_cloned!(self.selected_ion_mz),
            finish_cloned!(self.charge_state),
            finish_cloned!(self.intensity),
            finish_cloned!(self.ion_mobility),
            self.ion_mobility_type.finish_cloned(),
            self.parameters.finish_cloned(),
        ];

        for e in self.extra.iter() {
            arrays.push(e.finish_cloned());
        }

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    anyways!();
}

impl StructVisitor<(u64, u64, &mzdata::spectrum::SelectedIon)> for SelectedIonBuilder {
    fn append_value(&mut self, item: &(u64, u64, &mzdata::spectrum::SelectedIon)) -> bool {
        let (i, j, item) = item;
        self.spectrum_index.append_value(*i);
        self.precursor_index.append_value(*j);
        self.selected_ion_mz.append_value(item.mz);
        self.charge_state.append_option(item.charge());
        self.intensity.append_value(item.intensity);
        let im_curie = if let Some(im_val) = item.ion_mobility_type() {
            self.ion_mobility.append_value(im_val.to_f64().unwrap());
            let c = im_val.curie();
            self.ion_mobility_type.append_option(c.as_ref());
            c
        } else {
            self.ion_mobility.append_null();
            self.ion_mobility_type.append_null();
            None
        };

        let b = self.parameters.as_mut().values();
        for param in item.params() {
            if im_curie.is_some() {
                if param.curie() == im_curie {
                    continue;
                }
                b.append_value(param);
            } else {
                b.append_value(param);
            }
        }

        self.parameters.as_mut().append(true);

        for e in self.extra.iter_mut() {
            e.append_value(item);
        }

        true
    }
}

impl VisitorBase for SelectedIonBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = vec![
            field!("spectrum_index", DataType::UInt64),
            field!("precursor_index", DataType::UInt64),
            field!("selected_ion_mz", DataType::Float64),
            field!("charge_state", DataType::Int32),
            field!("intensity", DataType::Float32),
            field!("ion_mobility", DataType::Float64),
            field!("ion_mobility_type", self.ion_mobility_type.as_struct_type()),
        ];
        fields.extend(self.parameters.schema());
        for e in self.extra.iter() {
            fields.extend(e.schema());
        }
        fields
    }

    fn append_null(&mut self) {
        self.spectrum_index.append_null();
        self.precursor_index.append_null();
        self.selected_ion_mz.append_null();
        self.charge_state.append_null();
        self.intensity.append_null();
        self.ion_mobility.append_null();
        self.ion_mobility_type.append_null();
        self.parameters.append_null();
        for e in self.extra.iter_mut() {
            e.append_null();
        }
    }
}

#[derive(Debug)]
pub struct AuxiliaryArrayBuilder {
    data: LargeListBuilder<UInt8Builder>,
    name: ParamBuilder,
    data_type: CURIEBuilder,
    compression: CURIEBuilder,
    unit: CURIEBuilder,
    parameters: ParamListBuilder,
    data_processing_ref: LargeStringBuilder,
}

impl Default for AuxiliaryArrayBuilder {
    fn default() -> Self {
        Self {
            data: LargeListBuilder::new(UInt8Builder::new()).with_field(field!(
                "item",
                DataType::UInt8,
                false
            )),
            name: Default::default(),
            data_type: Default::default(),
            compression: Default::default(),
            unit: Default::default(),
            parameters: Default::default(),
            data_processing_ref: Default::default(),
        }
    }
}

impl VisitorBase for AuxiliaryArrayBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![
            field!(
                "data",
                DataType::LargeList(field!("item", DataType::UInt8, false))
            ),
            field!("name", self.name.as_struct_type()),
            field!("data_type", self.data_type.as_struct_type()),
            field!("unit", self.unit.as_struct_type()),
            field!(
                "parameters",
                DataType::LargeList(field!(
                    "item",
                    self.parameters.0.values_ref().as_struct_type()
                ))
            ),
            field!("data_processing_ref", DataType::LargeUtf8),
        ]
    }

    fn append_null(&mut self) {
        self.data.append_null();
        self.name.append_null();
        self.data_type.append_null();
        self.compression.append_null();
        self.unit.append_null();
        self.parameters.append_null();
        self.data_processing_ref.append_null();
    }
}

impl StructVisitor<AuxiliaryArray> for AuxiliaryArrayBuilder {
    fn append_value(&mut self, item: &AuxiliaryArray) -> bool {
        self.data.values().append_slice(&item.data);
        self.data.append(true);
        self.name.append_value(&item.name);
        self.data_type.append_value(&item.data_type);
        self.compression.append_option(item.unit.as_ref());
        self.parameters.append_value(&item.parameters.as_slice());
        self.data_processing_ref
            .append_option(item.data_processing_ref.as_ref());
        true
    }
}

impl ArrayBuilder for AuxiliaryArrayBuilder {
    anyways!();

    fn len(&self) -> usize {
        self.name.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();

        let arrays: Vec<ArrayRef> = vec![
            finish_it!(self.data),
            self.name.finish(),
            self.data_type.finish(),
            self.unit.finish(),
            finish_it!(self.parameters),
            finish_it!(self.data_processing_ref),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let fields = self.schema();

        let arrays: Vec<ArrayRef> = vec![
            finish_cloned!(self.data),
            self.name.finish_cloned(),
            self.data_type.finish_cloned(),
            self.unit.finish_cloned(),
            finish_cloned!(self.parameters),
            finish_cloned!(self.data_processing_ref),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }
}


#[derive(Debug)]
pub enum SpectrumVisitor {
    Description(Box<dyn StructVisitorBuilder<SpectrumDescription>>),
}

impl ArrayBuilder for SpectrumVisitor {
    anyways!();

    fn len(&self) -> usize {
        match self {
            Self::Description(builder) => builder.len()
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Description(builder) => builder.finish()
        }
    }

    fn finish_cloned(&self) -> ArrayRef {
        match self {
            Self::Description(builder) => builder.finish_cloned()
        }
    }
}

impl StructVisitor<SpectrumDescription> for SpectrumVisitor {
    fn append_value(&mut self, item: &SpectrumDescription) -> bool {
        match self {
            Self::Description(builder) => builder.append_value(item)
        }
    }
}

impl VisitorBase for SpectrumVisitor {
    fn schema(&self) -> Vec<FieldRef> {
        match self {
            SpectrumVisitor::Description(struct_visitor_builder) => struct_visitor_builder.schema(),
        }
    }

    fn append_null(&mut self) {
        match self {
            SpectrumVisitor::Description(struct_visitor_builder) => struct_visitor_builder.append_null(),
        }
    }
}


#[derive(Default, Debug)]
pub struct SpectrumDetailsBuilder {
    index: UInt64Builder,
    id: LargeStringBuilder,
    ms_level: UInt8Builder,
    time: Float32Builder,
    polarity: Int8Builder,
    mz_signal_continuity: CURIEBuilder,
    spectrum_type: CURIEBuilder,
    lowest_observed_mz: Float64Builder,
    highest_observed_mz: Float64Builder,
    lowest_observed_wavelength: Float64Builder,
    highest_observed_wavelength: Float64Builder,
    lowest_observed_ion_mobility: Float64Builder,
    highest_observed_ion_mobility: Float64Builder,
    number_of_data_points: UInt64Builder,
    base_peak_mz: Float64Builder,
    base_peak_intensity: Float32Builder,
    total_ion_current: Float32Builder,
    data_processing_ref: UInt32Builder,
    parameters: ParamListBuilder,
    auxiliary_arrays: LargeListBuilder<AuxiliaryArrayBuilder>,
    number_of_auxiliary_arrays: UInt32Builder,
    mz_delta_model: LargeListBuilder<Float64Builder>,
    extra: Vec<SpectrumVisitor>,
}

impl VisitorBase for SpectrumDetailsBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        let mut fields = vec![
            field!("index", DataType::UInt64),
            field!("id", DataType::LargeUtf8),
            field!("ms_level", DataType::UInt8),
            field!("time", DataType::Float32),
            field!("polarity", DataType::Int8),
            field!(
                "mz_signal_continuity",
                self.mz_signal_continuity.as_struct_type()
            ),
            field!("spectrum_type", self.spectrum_type.as_struct_type()),
            field!("lowest_observed_mz", DataType::Float64),
            field!("highest_observed_mz", DataType::Float64),
            field!("lowest_observed_wavelength", DataType::Float64),
            field!("highest_observed_wavelength", DataType::Float64),
            field!("lowest_observed_ion_mobility", DataType::Float64),
            field!("highest_observed_ion_mobility", DataType::Float64),
            field!("number_of_data_points", DataType::UInt64),
            field!("base_peak_mz", DataType::Float64),
            field!("base_peak_intensity", DataType::Float32),
            field!("total_ion_current", DataType::Float32),
            field!("data_processing_ref", DataType::UInt32),
            field!(
                "parameters",
                DataType::LargeList(field!(
                    "item",
                    self.parameters.0.values_ref().as_struct_type()
                ))
            ),
            field!(
                "auxiliary_arrays",
                DataType::LargeList(field!(
                    "item",
                    self.auxiliary_arrays.values_ref().as_struct_type()
                ))
            ),
            field!("number_of_auxiliary_arrays", DataType::UInt32),
            field!(
                "mz_delta_model",
                DataType::LargeList(field!("item", DataType::Float64))
            ),
        ];

        for e in self.extra.iter() {
            fields.extend(e.schema());
        }
        fields
    }

    fn append_null(&mut self) {
        self.index.append_null();
        self.id.append_null();
        self.ms_level.append_null();
        self.time.append_null();
        self.polarity.append_null();
        self.mz_signal_continuity.append_null();
        self.spectrum_type.append_null();
        self.lowest_observed_mz.append_null();
        self.highest_observed_mz.append_null();
        self.lowest_observed_wavelength.append_null();
        self.highest_observed_wavelength.append_null();
        self.lowest_observed_ion_mobility.append_null();
        self.highest_observed_ion_mobility.append_null();
        self.number_of_data_points.append_null();
        self.base_peak_mz.append_null();
        self.base_peak_intensity.append_null();
        self.total_ion_current.append_null();
        self.data_processing_ref.append_null();
        self.parameters.append_null();
        self.auxiliary_arrays.append_null();
        self.number_of_auxiliary_arrays.append_null();
        self.mz_delta_model.append_null();
        for e in self.extra.iter_mut() {
            e.append_null();
        }
    }
}

impl SpectrumDetailsBuilder {
    pub fn append_value<
        C: CentroidLike,
        D: DeconvolutedCentroidLike,
        S: SpectrumLike<C, D> + 'static,
    >(
        &mut self,
        index: u64,
        item: &S,
        mz_delta_model_params: Option<Vec<f64>>,
        auxiliary_arrays: Option<Vec<AuxiliaryArray>>,
    ) -> bool {
        let summaries = item.peaks().fetch_summaries();

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

        let spectrum_type = if let Some(v) = item
            .spectrum_type()
            .map(|t| crate::CURIE::from(t.to_param().curie().unwrap()))
        {
            v
        } else {
            match item.ms_level() {
                0 => panic!("Unsupported ms level"),
                1 => crate::curie!(MS:1000579),
                _ => crate::curie!(MS:1000580),
            }
        };

        self.index.append_value(index);
        self.id.append_value(item.id());
        self.ms_level.append_value(item.ms_level());
        self.time.append_value(item.start_time() as f32);
        self.polarity.append_value(match item.polarity() {
            ScanPolarity::Positive => 1,
            ScanPolarity::Negative => -1,
            ScanPolarity::Unknown => 0,
        });
        self.mz_signal_continuity
            .append_value(&match item.signal_continuity() {
                mzdata::spectrum::SignalContinuity::Unknown => crate::curie!(MS:1000525),
                mzdata::spectrum::SignalContinuity::Centroid => crate::curie!(MS:1000127),
                mzdata::spectrum::SignalContinuity::Profile => crate::curie!(MS:1000128),
            });

        self.spectrum_type.append_value(&spectrum_type);

        self.lowest_observed_mz.append_value(summaries.mz_range.0);
        self.highest_observed_mz.append_value(summaries.mz_range.1);

        self.lowest_observed_ion_mobility.append_null();
        self.highest_observed_ion_mobility.append_null();

        self.lowest_observed_wavelength.append_null();
        self.highest_observed_wavelength.append_null();

        self.base_peak_mz.append_option(base_peak_mz);
        self.base_peak_intensity.append_option(base_peak_intensity);
        self.total_ion_current.append_value(summaries.tic);

        self.parameters.append_value(&item.params());

        self.number_of_data_points.append_value(n_pts as u64);

        self.data_processing_ref.append_null();

        if let Some(arrays) = auxiliary_arrays.as_ref() {
            let b = self.auxiliary_arrays.values();
            for a in arrays {
                b.append_value(&a);
            }
            self.auxiliary_arrays.append(true);
        } else {
            self.auxiliary_arrays.append_null();
        }

        self.number_of_auxiliary_arrays
            .append_value(auxiliary_arrays.map(|v| v.len()).unwrap_or_default() as u32);

        match mz_delta_model_params {
            Some(params) => {
                self.mz_delta_model.values().append_slice(&params);
                self.mz_delta_model.append(true);
            }
            _ => {
                self.mz_delta_model.append_null();
            }
        };

        for e in self.extra.iter_mut() {
            eprintln!("\nEvaluating extra {e:?}");
            match e {
                SpectrumVisitor::Description(builder) => {
                    builder.append_value(item.description());
                },
            }
        }

        true
    }
}

impl ArrayBuilder for SpectrumDetailsBuilder {
    anyways!();

    fn len(&self) -> usize {
        self.index.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let schema = self.schema();

        let mut arrays: Vec<ArrayRef> = vec![
            finish_it!(self.index),
            finish_it!(self.id),
            finish_it!(self.ms_level),
            finish_it!(self.time),
            finish_it!(self.polarity),
            self.mz_signal_continuity.finish(),
            self.spectrum_type.finish(),
            finish_it!(self.lowest_observed_mz),
            finish_it!(self.highest_observed_mz),
            finish_it!(self.lowest_observed_wavelength),
            finish_it!(self.highest_observed_wavelength),
            finish_it!(self.lowest_observed_ion_mobility),
            finish_it!(self.highest_observed_ion_mobility),
            finish_it!(self.number_of_data_points),
            finish_it!(self.base_peak_mz),
            finish_it!(self.base_peak_intensity),
            finish_it!(self.total_ion_current),
            finish_it!(self.data_processing_ref),
            self.parameters.finish(),
            finish_it!(self.auxiliary_arrays),
            finish_it!(self.number_of_auxiliary_arrays),
            finish_it!(self.mz_delta_model),
        ];

        for e in self.extra.iter_mut() {
            arrays.push(e.finish());
        }
        Arc::new(StructArray::new(schema.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let schema = self.schema();

        let mut arrays: Vec<ArrayRef> = vec![
            finish_cloned!(self.index),
            finish_cloned!(self.id),
            finish_cloned!(self.ms_level),
            finish_cloned!(self.time),
            finish_cloned!(self.polarity),
            self.mz_signal_continuity.finish_cloned(),
            self.spectrum_type.finish_cloned(),
            finish_cloned!(self.lowest_observed_mz),
            finish_cloned!(self.highest_observed_mz),
            finish_cloned!(self.lowest_observed_wavelength),
            finish_cloned!(self.highest_observed_wavelength),
            finish_cloned!(self.lowest_observed_ion_mobility),
            finish_cloned!(self.highest_observed_ion_mobility),
            finish_cloned!(self.number_of_data_points),
            finish_cloned!(self.base_peak_mz),
            finish_cloned!(self.base_peak_intensity),
            finish_cloned!(self.total_ion_current),
            finish_cloned!(self.data_processing_ref),
            self.parameters.finish_cloned(),
            finish_cloned!(self.auxiliary_arrays),
            finish_cloned!(self.number_of_auxiliary_arrays),
            finish_cloned!(self.mz_delta_model),
        ];

        for e in self.extra.iter() {
            arrays.push(e.finish_cloned());
        }
        Arc::new(StructArray::new(schema.into(), arrays, None))
    }
}

#[derive(Default, Debug)]
pub struct SpectrumBuilder {
    spectrum_index_counter: u64,
    precursor_index_counter: u64,
    spectrum: SpectrumDetailsBuilder,
    scan: ScanBuilder,
    precursor: PrecursorBuilder,
    selected_ion: SelectedIonBuilder,
}

impl SpectrumBuilder {
    pub fn append_value<
        C: CentroidLike,
        D: DeconvolutedCentroidLike,
        S: SpectrumLike<C, D> + 'static,
    >(
        &mut self,
        item: &S,
        mz_delta_model_params: Option<Vec<f64>>,
        auxiliary_arrays: Option<Vec<AuxiliaryArray>>,
    ) -> bool {
        let out = self.spectrum.append_value(
            self.spectrum_index_counter,
            item,
            mz_delta_model_params,
            auxiliary_arrays,
        );
        for s in item.acquisition().scans.iter() {
            self.scan.append_value(&(self.spectrum_index_counter, s));
        }
        for precursor in item.precursor_iter() {
            self.precursor.append_value(&(
                self.spectrum_index_counter,
                self.precursor_index_counter,
                precursor,
            ));
            for ion in precursor.iter() {
                self.selected_ion.append_value(&(
                    self.spectrum_index_counter,
                    self.precursor_index_counter,
                    ion,
                ));
            }
            self.precursor_index_counter += 1;
        }
        self.spectrum_index_counter += 1;
        out
    }

    pub fn add_spectrum_param_field<T: StructVisitorBuilder<SpectrumDescription>>(
        &mut self,
        visitor: T,
    ) {
        self.spectrum.extra.push(SpectrumVisitor::Description(Box::new(visitor)));
    }

    pub fn add_selected_ion_field(
        &mut self,
        visitor: impl StructVisitorBuilder<mzdata::spectrum::SelectedIon>,
    ) {
        self.selected_ion.extra.push(Box::new(visitor));
    }

    pub fn add_scan_field(
        &mut self,
        visitor: impl StructVisitorBuilder<mzdata::spectrum::ScanEvent>,
    ) {
        self.scan.extra.push(Box::new(visitor));
    }
}

impl VisitorBase for SpectrumBuilder {
    fn schema(&self) -> Vec<FieldRef> {
        vec![
            field!("spectrum", self.spectrum.as_struct_type()),
            field!("scan", self.scan.as_struct_type()),
            field!("precursor", self.precursor.as_struct_type()),
            field!("selected_ion", self.selected_ion.as_struct_type()),
        ]
    }

    fn append_null(&mut self) {
        self.spectrum.append_null();
        self.scan.append_null();
        self.precursor.append_null();
        self.selected_ion.append_null();
    }
}

impl ArrayBuilder for SpectrumBuilder {
    anyways!();

    fn len(&self) -> usize {
        self.spectrum.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let fields = self.schema();
        let n = self
            .spectrum
            .len()
            .max(self.scan.len())
            .max(self.precursor.len())
            .max(self.selected_ion.len());

        while n > self.spectrum.len() {
            self.spectrum.append_null();
        }

        while n > self.scan.len() {
            self.scan.append_null();
        }

        while n > self.precursor.len() {
            self.precursor.append_null();
        }

        while n > self.selected_ion.len() {
            self.selected_ion.append_null();
        }

        let arrays = vec![
            self.spectrum.finish(),
            self.scan.finish(),
            self.precursor.finish(),
            self.selected_ion.finish(),
        ];

        Arc::new(StructArray::new(fields.into(), arrays, None))
    }

    fn finish_cloned(&self) -> ArrayRef {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use mzdata;
    use std::io;

    #[test]
    fn test_build_spectra() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let spec = reader.get_spectrum_by_index(2).unwrap();
        eprintln!("{:?}", spec.description());

        let mut builder = SpectrumBuilder::default();

        builder.add_spectrum_param_field(CustomBuilderFromParameter::from_spec(
            mzdata::curie!(MS:1000504),
            "base peak m/z",
            DataType::Float64,
        ));

        builder.append_value(&spec, None, None);
        let arrays = builder.finish();
        eprintln!("{arrays:#?}");
        Ok(())
    }
}
