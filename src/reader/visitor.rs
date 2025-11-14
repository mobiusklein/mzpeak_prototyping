use std::collections::HashMap;

use arrow::{
    array::{
        Array, ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int32Array,
        Int64Array, LargeListArray, LargeStringArray, StringArray, StructArray, UInt8Array,
        UInt32Array, UInt64Array,
    },
    buffer::NullBuffer,
    datatypes::{
        DataType, FieldRef, Float32Type, Float64Type, Int32Type, Int64Type, UInt8Type, UInt32Type,
        UInt64Type,
    },
};
use itertools::Itertools;
use mzdata::{
    curie,
    meta::SpectrumType,
    params::{ControlledVocabulary, Unit},
    prelude::*,
    spectrum::{
        ArrayType, BinaryDataArrayType, ChromatogramDescription, ChromatogramType, DataArray,
        Precursor, ScanEvent, ScanPolarity, ScanWindow, SelectedIon, SpectrumDescription,
        bindata::BinaryCompressionType,
    },
};

use crate::{CURIE, param::MetadataColumn};

pub fn parse_column_to_curie(column_name: &str) -> Option<(mzdata::params::CURIE, String)> {
    let mut it = column_name.split("_");
    let prefix = it.next()?;
    let cv: ControlledVocabulary = prefix.parse().ok()?;
    if matches!(cv, ControlledVocabulary::Unknown) {
        return None;
    }
    let accession = it.next()?;
    let ident = mzdata::params::CURIE::new(cv, accession.parse().ok()?);
    let name = it.join("_");
    Some((ident, name))
}

pub(crate) fn metadata_columns_to_definition_map(
    columns: Vec<MetadataColumn>,
) -> HashMap<String, MetadataColumn> {
    let mut table = HashMap::with_capacity(columns.len());
    for col in columns {
        let key = col.path.last().unwrap().clone();
        table.insert(key, col);
    }
    table
}

pub fn schema_to_metadata_cols<'a>(
    fields: impl IntoIterator<Item = &'a FieldRef>,
    prefix: String,
    defined_columns: Option<&HashMap<String, MetadataColumn>>,
) -> Vec<MetadataColumn> {
    let mut columns = Vec::new();
    for (i, f) in fields.into_iter().enumerate() {
        if let Some((curie, mangled_name)) = parse_column_to_curie(f.name()) {
            log::trace!("Adding parsed {curie}|{mangled_name} @ {i} from {prefix}");
            columns.push(MetadataColumn::new(
                mangled_name,
                vec![prefix.clone(), f.name().to_string()],
                i,
                Some(curie.into()),
            ))
        } else if let Some(defined) = defined_columns {
            if let Some(defined_col) = defined.get(f.name()) {
                log::trace!(
                    "Adding defined {1}|{0} @ {i} from {prefix}",
                    defined_col.name,
                    defined_col.accession.unwrap()
                );
                let mut col = defined_col.clone();
                col.path[0] = prefix.clone();
                col.index = i;
                columns.push(col);
            }
        }
    }
    columns
}

pub struct ParameterVisitor<'a> {
    root: &'a StructArray,
    destination: Vec<mzdata::Param>,
}

impl<'a> ParameterVisitor<'a> {
    pub fn new(root: &'a StructArray) -> Self {
        Self {
            root,
            destination: Vec::new(),
        }
    }

    pub fn build(mut self) -> Vec<mzdata::Param> {
        let n = self.root.len();
        self.destination.resize(n, Default::default());

        if let Some(name) = self.root.column_by_name("name") {
            let arr = name.as_string::<i64>();
            for (i, v) in arr.iter().enumerate() {
                self.destination[i].name = v.unwrap().to_string();
            }
        }
        if let Some(curie) = self
            .root
            .column_by_name("curie")
            .or_else(|| self.root.column_by_name("accession"))
        {
            let arr = AnyCURIEArray::try_from(curie).unwrap();
            for i in 0..n {
                if let Some(v) = arr.value(i).map(mzdata::params::CURIE::from) {
                    let d = &mut self.destination[i];
                    d.accession = Some(v.accession);
                    d.controlled_vocabulary = Some(v.controlled_vocabulary);
                }
            }
        }
        if let Some(unit) = self.root.column_by_name("unit") {
            let arr = UnitArray::from(unit);
            for i in 0..n {
                self.destination[i].unit = arr.value(i);
            }
        }
        if let Some(values) = self.root.column_by_name("value") {
            let values = values.as_struct();
            if let Some(ints) = values.column_by_name("integer") {
                if ints.null_count() != n {
                    let ints = ints.as_primitive::<Int64Type>();
                    for i in 0..n {
                        if ints.is_valid(i) {
                            self.destination[i].value = mzdata::params::Value::Int(ints.value(i));
                        }
                    }
                }
            }
            if let Some(ints) = values.column_by_name("float") {
                if ints.null_count() != n {
                    let ints = ints.as_primitive::<Float64Type>();
                    for i in 0..n {
                        if ints.is_valid(i) {
                            self.destination[i].value = mzdata::params::Value::Float(ints.value(i));
                        }
                    }
                }
            }
            if let Some(ints) = values.column_by_name("string") {
                if ints.null_count() != n {
                    let ints = ints.as_string::<i64>();
                    for i in 0..n {
                        if ints.is_valid(i) {
                            self.destination[i].value =
                                mzdata::params::Value::String(ints.value(i).to_string());
                        }
                    }
                }
            }
            if let Some(ints) = values.column_by_name("boolean") {
                if ints.null_count() != n {
                    let ints = ints.as_boolean();
                    for i in 0..n {
                        if ints.is_valid(i) {
                            self.destination[i].value =
                                mzdata::params::Value::Boolean(ints.value(i));
                        }
                    }
                }
            }
        }
        self.destination
    }
}

pub(crate) struct MzSpectrumVisitor<'a> {
    pub(crate) descriptions: &'a mut [SpectrumDescription],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

macro_rules! extract_unit {
    ($metacol:ident, $array:ident) => {{
        let units: UnitCollection<'_> = match &$metacol.unit {
            crate::param::PathOrCURIE::Path(items) => {
                if let Some(val) = $array.column_by_name(items.last().unwrap()) {
                    UnitCollection::series(val.as_struct())
                } else {
                    log::error!("The path {} did not exist", items.join("."));
                    UnitCollection::unknown()
                }
            }
            crate::param::PathOrCURIE::CURIE(curie) => {
                UnitCollection::singular(Unit::from_curie(&(*curie).into()))
            }
            crate::param::PathOrCURIE::None => UnitCollection::unknown(),
        };
        units
    }};
}

impl<'a> VisitorBuilderBase<'a, SpectrumDescription> for MzSpectrumVisitor<'a> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, SpectrumDescription> {
        OffsetCollection::new(self.descriptions, &self.offsets)
    }

    fn metadata_map(&self) -> &'a [MetadataColumn] {
        &self.metadata_map
    }
}

impl<'a> VisitorBuilder1<'a, SpectrumDescription> for MzSpectrumVisitor<'a> {}

impl<'a> MzSpectrumVisitor<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [SpectrumDescription],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets: Vec::new(),
        }
    }

    fn visit_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, descr) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            descr.index = val as usize;
        }
        self.offsets = offsets
    }

    fn visit_id(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &LargeStringArray = spec_arr.column(index).as_string();
        for (i, descr) in self.iter_instances() {
            let val = arr.value(i);
            descr.id = val.to_string();
        }
    }

    fn visit_ms_level(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt8Type>();
        for (i, descr) in self.iter_instances() {
            if arr.is_null(i) {
                continue;
            };
            let ms_level_val = arr.value(i);
            descr.ms_level = ms_level_val;
        }
    }

    fn visit_polarity(&mut self, spec_arr: &StructArray, index: usize) {
        let polarity_arr: &Int8Array = spec_arr.column(index).as_any().downcast_ref().unwrap();
        for (i, descr) in self.iter_instances() {
            let polarity_val = polarity_arr.value(i);
            match polarity_val {
                1 => descr.polarity = ScanPolarity::Positive,
                -1 => descr.polarity = ScanPolarity::Negative,
                _ => {
                    todo!("Don't know how to deal with polarity {polarity_val}")
                }
            }
        }
    }

    fn visit_mz_signal_continuity(&mut self, spec_arr: &StructArray, index: usize) {
        let continuity_array = AnyCURIEArray::try_from(spec_arr.column(index)).unwrap();

        for (i, descr) in self.iter_instances() {
            if continuity_array.is_null(i) {
                continue;
            };
            let continuity_curie = continuity_array.value(i).unwrap();
            descr.signal_continuity = match continuity_curie {
                curie!(MS:1000525) => mzdata::spectrum::SignalContinuity::Unknown,
                curie!(MS:1000127) => mzdata::spectrum::SignalContinuity::Centroid,
                curie!(MS:1000128) => mzdata::spectrum::SignalContinuity::Profile,
                _ => todo!("Don't know how to deal with {continuity_curie}"),
            };
        }
    }

    fn visit_spectrum_type(&mut self, spec_arr: &StructArray, index: usize) {
        let spec_type_array = spec_arr.column(index);

        let curie_array = AnyCURIEArray::try_from(spec_type_array).unwrap();

        for (i, descr) in self.iter_instances() {
            let spec_type = curie_array
                .value(i)
                .and_then(|v| SpectrumType::from_accession(v.accession));
            if let Some(spec_type) = spec_type {
                descr.set_spectrum_type(spec_type);
            }
        }
    }

    fn visit_lowest_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances()
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000528))
                        .name("lowest observed m/z")
                        .unit(Unit::MZ)
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?} for lowest m/z", arr.data_type())
        }
    }

    fn visit_highest_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances()
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000527))
                        .name("highest observed m/z")
                        .unit(Unit::MZ)
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?} for lowest m/z", arr.data_type())
        }
    }

    fn visit_base_peak_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances()
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000504))
                        .name("base peak m/z")
                        .unit(Unit::MZ)
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?} for lowest m/z", arr.data_type())
        }
    }

    fn visit_base_peak_intensity(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: &MetadataColumn,
    ) {
        let unit = extract_unit!(metacol, spec_arr);

        macro_rules! extract {
            ($arr:expr) => {
                for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
                    if $arr.is_null(i) { continue };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000505))
                        .name("base peak intensity")
                        .unit(unit.value(i))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            extract!(arr);
        } else {
            unimplemented!("{:?} not supported for {metacol:?}", arr.data_type())
        }
    }

    fn visit_total_ion_current(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: &MetadataColumn,
    ) {
        let unit = extract_unit!(metacol, spec_arr);
        macro_rules! extract {
            ($arr:expr) => {
                for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
                    if $arr.is_null(i) { continue };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000285))
                        .name("total ion current")
                        .unit(unit.value(i))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);

        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            extract!(arr);
        } else {
            unimplemented!("{:?} not supported for {metacol:?}", arr.data_type())
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) -> usize {
        let names = spec_arr.column_names();
        let mut visited = vec![false; spec_arr.num_columns()];

        // Must visit the index first, to infer null spacing
        if let Some(i) = names.iter().position(|v| *v == "index") {
            self.visit_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Spectrum arrays did not contain \"index\" column");
            panic!("Spectrum arrays did not contain \"index\" column: {names:?}");
        }

        if let Some(i) = names.iter().position(|v| *v == "id") {
            self.visit_id(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Spectrum arrays did not contain \"id\" column");
        }

        for col in self.metadata_map.iter() {
            log::trace!("Visiting spectrum {col:?}");
            if let Some(accession) = col.accession {
                match accession {
                    curie!(MS:1000511) => {
                        self.visit_ms_level(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000465) => {
                        self.visit_polarity(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000525) => {
                        self.visit_mz_signal_continuity(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000559) => {
                        self.visit_spectrum_type(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000528) => {
                        self.visit_lowest_mz(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000527) => {
                        self.visit_highest_mz(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000504) => {
                        self.visit_base_peak_mz(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000505) => {
                        self.visit_base_peak_intensity(spec_arr, col.index, col);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000285) => {
                        self.visit_total_ion_current(spec_arr, col.index, col);
                        visited[col.index] = true;
                    }
                    curie!(MS:1003060) => {
                        // number of data points
                        visited[col.index] = true;
                    }
                    _ => {
                        self.visit_as_param(spec_arr, col.index, col);
                        visited[col.index] = true;
                    }
                }
            }
        }
        const SKIP_PARAMS: [CURIE; 6] = [
            curie!(MS:1000505),
            curie!(MS:1000504),
            curie!(MS:1000257),
            curie!(MS:1000285),
            curie!(MS:1000527),
            curie!(MS:1000528),
        ];

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            log::trace!("Visiting spectrum {colname} ({index})");
            match colname {
                "polarity" => self.visit_polarity(spec_arr, index),
                "spectrum_type" => self.visit_spectrum_type(spec_arr, index),
                "mz_signal_continuity" => self.visit_mz_signal_continuity(spec_arr, index),
                "ms_level" => self.visit_ms_level(spec_arr, index),
                "lowest_observed_mz" => self.visit_lowest_mz(spec_arr, index),
                "highest_observed_mz" => self.visit_highest_mz(spec_arr, index),
                "number_of_data_points" => {}
                "base_peak_mz" => self.visit_base_peak_mz(spec_arr, index),
                "base_peak_intensity" => self.visit_base_peak_intensity(
                    spec_arr,
                    index,
                    &MetadataColumn {
                        name: "".into(),
                        path: vec![],
                        index: 0,
                        accession: None,
                        unit: Unit::DetectorCounts.into(),
                    },
                ),
                "total_ion_current" => {
                    self.visit_total_ion_current(
                        spec_arr,
                        index,
                        &MetadataColumn {
                            name: "".into(),
                            path: vec![],
                            index: 0,
                            accession: None,
                            unit: Unit::DetectorCounts.into(),
                        },
                    );
                }
                "parameters" => {
                    self.visit_parameters(spec_arr, &SKIP_PARAMS);
                }
                "data_processing_ref" => {
                    // TODO: Add a slot for this in the `mzdata` data model
                }
                _ => {}
            }
        }

        return self.offsets.len();
    }
}

pub struct CURIEStrArray<'a>(&'a StringArray);

impl<'a> From<&'a StringArray> for CURIEStrArray<'a> {
    fn from(value: &'a StringArray) -> Self {
        Self(value)
    }
}

impl<'a> CURIEStrArray<'a> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_null(&self, index: usize) -> bool {
        self.0.is_null(index)
    }

    #[inline(always)]
    pub fn value(&self, index: usize) -> Option<mzdata::params::CURIE> {
        if self.is_null(index) {
            None
        } else {
            Some(self.0.value(index).parse().unwrap())
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Option<mzdata::params::CURIE>> {
        self.0.iter().map(|v| v.map(|v| v.parse().unwrap()))
    }
}

/// A deprecated struct-based encoding. To be removed before
/// first stable release.
pub struct CURIEStructArray<'a> {
    cv_id: &'a UInt8Array,
    accession: &'a UInt32Array,
    null: Option<&'a NullBuffer>,
}

impl<'a> From<&'a StructArray> for CURIEStructArray<'a> {
    fn from(value: &'a StructArray) -> Self {
        Self::from_struct_array(value)
    }
}

impl<'a> CURIEStructArray<'a> {
    fn new(
        cv_id: &'a UInt8Array,
        accession: &'a UInt32Array,
        null: Option<&'a NullBuffer>,
    ) -> Self {
        Self {
            cv_id,
            accession,
            null,
        }
    }

    pub fn len(&self) -> usize {
        self.cv_id.len()
    }

    pub fn from_struct_array(series: &'a StructArray) -> Self {
        Self::new(
            series.column(0).as_primitive(),
            series.column(1).as_primitive(),
            series.nulls(),
        )
    }

    #[inline(always)]
    pub fn is_null(&self, index: usize) -> bool {
        self.cv_id.is_null(index)
            || self.accession.is_null(index)
            || self.null.is_some_and(|v| v.is_null(index))
    }

    #[inline(always)]
    pub fn value(&self, index: usize) -> Option<CURIE> {
        if self.is_null(index) {
            None
        } else {
            Some(CURIE::new(
                match self.cv_id.value(index) {
                    1 => ControlledVocabulary::MS,
                    2 => ControlledVocabulary::UO,
                    x => panic!("Bad old CV ID {x}"),
                },
                self.accession.value(index),
            ))
        }
    }
}

pub enum AnyCURIEArray<'a> {
    Struct(CURIEStructArray<'a>),
    String(CURIEStrArray<'a>),
}

impl<'a> TryFrom<&'a ArrayRef> for AnyCURIEArray<'a> {
    type Error = ();

    fn try_from(value: &'a ArrayRef) -> Result<Self, Self::Error> {
        if let Some(arr) = value.as_struct_opt() {
            Ok(Self::from_struct_array(arr))
        } else {
            Ok(Self::String(CURIEStrArray(value.as_string::<i32>())))
        }
    }
}

impl<'a> AnyCURIEArray<'a> {
    pub fn from_struct_array(array: &'a StructArray) -> Self {
        Self::Struct(CURIEStructArray::from_struct_array(array))
    }

    pub fn len(&self) -> usize {
        match self {
            AnyCURIEArray::Struct(curiearray) => curiearray.len(),
            AnyCURIEArray::String(curiestr_array) => curiestr_array.len(),
        }
    }

    pub fn to_vec(&self) -> Vec<mzdata::params::CURIE> {
        match self {
            AnyCURIEArray::Struct(curiearray) => {
                let n = curiearray.len();
                let mut acc: Vec<_> = Vec::with_capacity(n);
                for i in 0..n {
                    acc.push(curiearray.value(i).unwrap().into())
                }
                acc
            }
            AnyCURIEArray::String(curiestr_array) => {
                let n = curiestr_array.len();
                let mut acc: Vec<_> = Vec::with_capacity(n);
                for i in 0..n {
                    acc.push(curiestr_array.value(i).unwrap())
                }
                acc
            }
        }
    }

    #[inline(always)]
    pub fn is_null(&self, index: usize) -> bool {
        match self {
            AnyCURIEArray::Struct(curiearray) => curiearray.is_null(index),
            AnyCURIEArray::String(curiestr_array) => curiestr_array.is_null(index),
        }
    }

    #[inline(always)]
    pub fn value(&self, index: usize) -> Option<CURIE> {
        match self {
            AnyCURIEArray::Struct(curiearray) => curiearray.value(index),
            AnyCURIEArray::String(curiestr_array) => curiestr_array.value(index).map(|v| v.into()),
        }
    }
}

impl<'a> From<CURIEStructArray<'a>> for AnyCURIEArray<'a> {
    fn from(value: CURIEStructArray<'a>) -> Self {
        Self::Struct(value)
    }
}

impl<'a> From<CURIEStrArray<'a>> for AnyCURIEArray<'a> {
    fn from(value: CURIEStrArray<'a>) -> Self {
        Self::String(value)
    }
}

pub struct UnitArray<'a>(AnyCURIEArray<'a>);

impl<'a> From<&'a ArrayRef> for UnitArray<'a> {
    fn from(value: &'a ArrayRef) -> Self {
        Self(AnyCURIEArray::try_from(value).unwrap())
    }
}

impl<'a> From<CURIEStructArray<'a>> for UnitArray<'a> {
    fn from(value: CURIEStructArray<'a>) -> Self {
        Self(value.into())
    }
}

impl<'a> From<&'a StructArray> for UnitArray<'a> {
    fn from(value: &'a StructArray) -> Self {
        Self(AnyCURIEArray::from_struct_array(value))
    }
}

impl<'a> UnitArray<'a> {
    pub fn from_struct_array(unit_series: &'a StructArray) -> Self {
        Self(AnyCURIEArray::from_struct_array(unit_series))
    }

    #[inline(always)]
    pub fn value(&self, index: usize) -> Unit {
        self.0
            .value(index)
            .map(|v| Unit::from_curie(&(v.into())))
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_null(&self, index: usize) -> bool {
        self.0.is_null(index)
    }
}

/// A single unit that is used for all values in a column
struct UnitScalar(Unit);

impl UnitScalar {
    #[inline(always)]
    fn value(&self, _index: usize) -> Unit {
        self.0
    }
}

/// A generic strategy for mapping units across rows of another column
enum UnitCollection<'a> {
    Array(UnitArray<'a>),
    Scalar(UnitScalar),
}

impl<'a> Default for UnitCollection<'a> {
    fn default() -> Self {
        Self::unknown()
    }
}

impl<'a> UnitCollection<'a> {
    #[inline(always)]
    fn value(&self, index: usize) -> Unit {
        match self {
            UnitCollection::Array(unit_series) => unit_series.value(index),
            UnitCollection::Scalar(unit_series_singular) => unit_series_singular.value(index),
        }
    }

    fn series(unit_series: &'a StructArray) -> Self {
        Self::Array(UnitArray::from_struct_array(unit_series))
    }

    fn singular(unit: Unit) -> Self {
        Self::Scalar(UnitScalar(unit))
    }

    fn unknown() -> Self {
        Self::Scalar(UnitScalar(Unit::Unknown))
    }
}

/// A type alias over a tuple of (primary source id, <entity>)
pub(crate) type Indexed<T> = (u64, T);
/// A type alias over a tuple of (primary source id, optional secondary id, <entity>)
pub(crate) type DoubleIndexed<T> = (u64, Option<u64>, T);

/// A helper trait for handling (multi-)-relationship keyed visitors over type `T`
pub(crate) trait CompoundIndexVisitor<T> {
    /// The primary source entity's id key
    fn source_index_mut(&mut self) -> &mut u64;

    /// The (potentially absent) secondary id key
    fn secondary_index_mut(&mut self) -> Option<&mut u64>;

    /// Get a mutable access to the actual thing we are visiting to build
    fn description_mut(&mut self) -> &mut T;

    /// Take the entity out of the container
    fn into_description(self) -> T;

    /// The primary source entity's id key
    fn source_index(&self) -> u64;

    /// The (potentially absent) secondary id key
    fn secondary_index(&self) -> Option<u64>;

    /// Unpack the container into a tuple of id keys and the entity description
    fn unpack(self) -> (u64, Option<u64>, T)
    where
        Self: Sized,
    {
        (
            self.source_index(),
            self.secondary_index(),
            self.into_description(),
        )
    }
}

impl<T> CompoundIndexVisitor<T> for Indexed<T> {
    fn description_mut(&mut self) -> &mut T {
        &mut self.1
    }

    fn source_index_mut(&mut self) -> &mut u64 {
        &mut self.0
    }

    fn secondary_index_mut(&mut self) -> Option<&mut u64> {
        None
    }

    fn into_description(self) -> T {
        self.1
    }

    fn source_index(&self) -> u64 {
        self.0
    }

    fn secondary_index(&self) -> Option<u64> {
        None
    }
}

impl<T> CompoundIndexVisitor<T> for DoubleIndexed<T> {
    fn description_mut(&mut self) -> &mut T {
        &mut self.2
    }

    fn source_index_mut(&mut self) -> &mut u64 {
        &mut self.0
    }

    fn secondary_index_mut(&mut self) -> Option<&mut u64> {
        self.1.as_mut()
    }

    fn into_description(self) -> T {
        self.2
    }

    fn source_index(&self) -> u64 {
        self.0
    }

    fn secondary_index(&self) -> Option<u64> {
        self.1
    }
}

/// Enclose the parallel arrays of "descriptions" and their offsets so that the borrow
/// checker knows that method calls on this instance aren't tied to the owning objects
struct OffsetCollection<'a, T> {
    pub(crate) descriptions: &'a mut [T],
    pub(crate) offsets: &'a [usize],
}

impl<'a, T> IntoIterator for OffsetCollection<'a, T> {
    type Item = (usize, &'a mut T);

    type IntoIter =
        std::iter::Zip<std::iter::Copied<std::slice::Iter<'a, usize>>, std::slice::IterMut<'a, T>>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.offsets
            .into_iter()
            .copied()
            .zip(self.descriptions.into_iter())
    }
}

impl<'a, T> OffsetCollection<'a, T> {
    fn new(descriptions: &'a mut [T], offsets: &'a [usize]) -> Self {
        Self {
            descriptions,
            offsets,
        }
    }
}

trait VisitorBuilderBase<'a, T> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, T>;
    #[allow(unused)]
    fn metadata_map(&self) -> &'a [MetadataColumn];
}

trait VisitorBuilder1<'a, T: ParamDescribed>: VisitorBuilderBase<'a, T> {
    fn visit_parameters(&mut self, struct_arr: &StructArray, skip_params: &[CURIE]) {
        let params_array: &LargeListArray =
            struct_arr.column_by_name("parameters").unwrap().as_list();

        for (i, descr) in self.iter_instances() {
            let params = params_array.value(i);
            let params = ParameterVisitor::new(params.as_struct()).build();

            for p in params {
                if let Some(acc) = p.curie().map(CURIE::from) {
                    if !skip_params.contains(&acc) {
                        descr.add_param(p.into());
                    }
                } else {
                    descr.add_param(p.into());
                }
            }
        }
    }

    fn visit_as_param(&mut self, spec_arr: &StructArray, index: usize, metacol: &MetadataColumn) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let units = extract_unit!(metacol, spec_arr);
        let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());

        macro_rules! convert {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances() {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p
                        .name(&metacol.name)
                        .value($arr.value(i))
                        .unit(units.value(i));
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }
}

trait VisitorBuilder2<'a, T: ParamDescribed>: VisitorBuilderBase<'a, Indexed<T>>
where
    Indexed<T>: CompoundIndexVisitor<T>,
{
    fn visit_as_param(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: Option<&MetadataColumn>,
        name: Option<&str>,
    ) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let (name, unit, accession) = if let Some(metacol) = metacol {
            let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());
            let units = extract_unit!(metacol, spec_arr);
            (metacol.name.as_str(), units, accession)
        } else if let Some(name) = name {
            (name, UnitCollection::unknown(), None)
        } else {
            panic!("One of `metacol` or `name` must be defined")
        };

        macro_rules! convert {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances() {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&name).value($arr.value(i)).unit(unit.value(i));
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.description_mut().add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray) {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();

        for (i, (_, descr)) in self.iter_instances() {
            let params = params_array.value(i);
            let params = params.as_struct();

            let params = ParameterVisitor::new(params).build();

            for p in params {
                descr.add_param(p);
            }
        }
    }
}

trait VisitorBuilder3<'a, T>: VisitorBuilderBase<'a, DoubleIndexed<T>>
where
    DoubleIndexed<T>: CompoundIndexVisitor<T>,
{
    fn visit_as_param(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: Option<&MetadataColumn>,
        name: Option<&str>,
    ) where
        T: ParamDescribed,
    {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let (name, unit, accession) = if let Some(metacol) = metacol {
            let unit = extract_unit!(metacol, spec_arr);
            let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());
            (metacol.name.as_str(), unit, accession)
        } else if let Some(name) = name {
            (name, Default::default(), None)
        } else {
            panic!("One of `metacol` or `name` must be defined")
        };

        macro_rules! convert {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances() {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&name).value($arr.value(i)).unit(unit.value(i));
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.description_mut().add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray)
    where
        T: ParamDescribed,
    {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();

        for (i, descr) in self.iter_instances() {
            let descr = descr.description_mut();
            let params = params_array.value(i);
            let params = params.as_struct();

            let params = ParameterVisitor::new(params).build();

            for p in params {
                descr.add_param(p);
            }
        }
    }

    fn visit_precursor_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        for (i, descr) in self.iter_instances() {
            if arr.is_null(i) {
                continue;
            };
            let mut v = arr.value(i);
            let _ = descr.secondary_index_mut().insert(&mut v);
        }
    }
}

pub(crate) struct MzScanVisitor<'a> {
    pub(crate) descriptions: &'a mut [Indexed<ScanEvent>],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> VisitorBuilderBase<'a, Indexed<ScanEvent>> for MzScanVisitor<'a> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, Indexed<ScanEvent>> {
        OffsetCollection::new(self.descriptions, &self.offsets)
    }

    fn metadata_map(&self) -> &'a [MetadataColumn] {
        self.metadata_map
    }
}

impl<'a> VisitorBuilder2<'a, ScanEvent> for MzScanVisitor<'a> {}

impl<'a> MzScanVisitor<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [(u64, ScanEvent)],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
        offsets: Vec<usize>,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets,
        }
    }

    fn visit_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, (index, _descr)) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            *index = val;
        }
        self.offsets = offsets
    }

    fn visit_instrument_configuration_ref(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.instrument_configuration_id = $arr.value(i) as u32;
                }
            };
        }

        let arr = spec_arr.column(index);

        if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_preset_scan_configuration(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .name("preset scan configuration")
                        .curie(mzdata::curie!(MS:1000616))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_filter_string(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .name("filter string")
                        .curie(mzdata::curie!(MS:1000512))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        if let Some(arr) = arr.as_string_opt::<i32>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_string_opt::<i64>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_scan_start_time(&mut self, spec_arr: &StructArray, index: usize, unit: Option<Unit>) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        let unit = unit.unwrap_or(Unit::Minute);
        let scalar = match unit {
            Unit::Minute => 1.0,
            Unit::Second => 1.0 / 60.0,
            Unit::Millisecond => 1.0 / (60.0 * 1000.0),
            _ => {
                log::error!("A unit {unit} other than a time unit provided, defaulting to minutes");
                1.0
            }
        };
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.start_time = $arr.value(i) as f64 * scalar;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_injection_time(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.injection_time = $arr.value(i) as f32;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_scan_windows_inner(scan_window_array: &StructArray) -> Vec<ScanWindow> {
        let lower_limit_arr = scan_window_array.column(0);
        let upper_limit_arr = scan_window_array.column(1);
        // let unit_arr = scan_window_array.column(2);
        // let params_arr = scan_window_array.column(3);
        let n = lower_limit_arr.len();
        let mut windows: Vec<ScanWindow> = Vec::with_capacity(n);
        windows.resize(n, Default::default());

        macro_rules! pack {
            ($dtype:ty) => {
                let lower_limit_arr = lower_limit_arr.as_primitive::<$dtype>();
                let upper_limit_arr = upper_limit_arr.as_primitive::<$dtype>();
                for (i, window) in windows.iter_mut().enumerate() {
                    window.lower_bound = lower_limit_arr.value(i) as f32;
                    window.upper_bound = upper_limit_arr.value(i) as f32;
                }
            };
        }

        if matches!(lower_limit_arr.data_type(), DataType::Float32) {
            pack!(Float32Type);
        } else if matches!(lower_limit_arr.data_type(), DataType::Float64) {
            pack!(Float64Type);
        } else {
            todo!("{:?} is not implemented", lower_limit_arr.data_type());
        }
        windows
    }

    fn visit_scan_windows(&mut self, spec_arr: &StructArray, index: usize) {
        // let fields: Vec<FieldRef> =
        //     SchemaLike::from_type::<ScanWindowEntry>(Default::default()).unwrap();

        let arr = spec_arr.column(index);

        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    let windows = $arr.value(i);
                    let windows = windows.as_struct();
                    descr.scan_windows = Self::visit_scan_windows_inner(windows);
                    // let windows: Vec<ScanWindowEntry> =
                    //     serde_arrow::from_arrow(&fields, windows.columns()).unwrap();
                    // for w in windows {
                    //     descr.scan_windows.push((&w).into());
                    // }
                }
            };
        }

        if let Some(arr) = arr.as_list_opt::<i64>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_list_opt::<i32>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type())
        }
    }

    fn visit_ion_mobility(
        &mut self,
        spec_arr: &StructArray,
        ion_mobility_value_index: usize,
        ion_mobility_type_index: usize,
    ) {
        let arr = spec_arr.column(ion_mobility_value_index);
        if arr.null_count() == arr.len() {
            return;
        }
        let ion_mobility_types = spec_arr.column(ion_mobility_type_index);
        let ion_mobility_types = AnyCURIEArray::try_from(ion_mobility_types).unwrap();

        macro_rules! pack {
            ($arr:ident) => {
                        for (i, (_, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if $arr.is_null(i) {
                continue;
            };
            let im_val = $arr.value(i);
            let im_tp = ion_mobility_types.value(i).unwrap();
            match im_tp {
                // ion mobility drift time
                curie!(MS:1002476) => descr.add_param(
                    mzdata::Param::builder()
                        .name("ion mobility drift time")
                        .curie(mzdata::curie!(MS:1002476))
                        .value(im_val)
                        .unit(Unit::Millisecond)
                        .build(),
                ),
                // inverse reduced ion mobility drift time
                curie!(MS:1002815) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("inverse reduced ion mobility drift time")
                            .curie(mzdata::curie!(MS:1002815))
                            .value(im_val)
                            .unit(Unit::VoltSecondPerSquareCentimeter)
                            .build()
                    )
                }
                // FAIMS compensation voltage
                curie!(MS:1001581) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("FAIMS compensation voltage")
                            .curie(mzdata::curie!(MS:1001581))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                // SELEXION compensation voltage
                curie!(MS:1003371) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("SELEXION compensation voltage")
                            .curie(mzdata::curie!(MS:1003371))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                _ => todo!("{im_tp} not supported yet"),
            }
        }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            todo!("{:?} not supported for ion mobility", arr.data_type());
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        let names = spec_arr.column_names();
        let mut visited = vec![false; spec_arr.num_columns()];
        // Must visit the index first, to infer null spacing
        if let Some(i) = names
            .iter()
            .position(|v| *v == "spectrum_index" || *v == "source_index")
        {
            self.visit_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Scan arrays did not contain \"index\" column");
            panic!("Scan arrays did not contain \"index\" column: {names:?}");
        }

        for col in self.metadata_map() {
            log::trace!("Visiting scan {col:?}");
            if let Some(accession) = col.accession {
                match accession {
                    curie!(MS:1000512) => {
                        self.visit_filter_string(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000616) => {
                        self.visit_preset_scan_configuration(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000016) => {
                        let unit = match &col.unit {
                            crate::param::PathOrCURIE::Path(_items) => todo!(),
                            crate::param::PathOrCURIE::CURIE(curie) => {
                                Some(Unit::from_curie(&((*curie).into())))
                            }
                            crate::param::PathOrCURIE::None => None,
                        };
                        self.visit_scan_start_time(spec_arr, col.index, unit);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000927) => {
                        self.visit_injection_time(spec_arr, col.index);
                        visited[col.index] = true;
                    }
                    _ => {
                        self.visit_as_param(spec_arr, col.index, Some(col), None);
                        visited[col.index] = true;
                    }
                }
            }
        }

        let mut ion_mobility_value_index = None;
        let mut ion_mobility_type_index = None;

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            log::trace!("Visiting scan {colname} ({index})");
            match colname {
                "parameters" => {
                    self.visit_parameters(spec_arr);
                }
                "instrument_configuration_ref" => {
                    self.visit_instrument_configuration_ref(spec_arr, index);
                }
                "scan_windows" => {
                    self.visit_scan_windows(spec_arr, index);
                }
                "ion_mobility_value" => {
                    ion_mobility_value_index = Some(index);
                }
                "ion_mobility_type" => {
                    ion_mobility_type_index = Some(index);
                }
                "scan_start_time" => {
                    self.visit_scan_start_time(spec_arr, index, None);
                }
                "injection_time" => {
                    self.visit_injection_time(spec_arr, index);
                }
                "filter_string" => {
                    self.visit_filter_string(spec_arr, index);
                }
                "preset_scan_configuration" => {
                    self.visit_preset_scan_configuration(spec_arr, index);
                }
                _ => {
                    self.visit_as_param(spec_arr, index, None, Some(colname));
                }
            }
        }

        match (ion_mobility_value_index, ion_mobility_type_index) {
            (Some(ion_mobility_value_index), Some(ion_mobility_type_index)) => {
                self.visit_ion_mobility(
                    spec_arr,
                    ion_mobility_value_index,
                    ion_mobility_type_index,
                );
            }
            (_, _) => {}
        }
    }
}

pub(crate) struct MzPrecursorVisitor<'a> {
    pub(crate) descriptions: &'a mut [DoubleIndexed<Precursor>],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> VisitorBuilderBase<'a, DoubleIndexed<Precursor>> for MzPrecursorVisitor<'a> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, DoubleIndexed<Precursor>> {
        OffsetCollection::new(self.descriptions, &self.offsets)
    }

    fn metadata_map(&self) -> &'a [MetadataColumn] {
        self.metadata_map
    }
}

impl<'a> VisitorBuilder3<'a, Precursor> for MzPrecursorVisitor<'a> {}

impl<'a> MzPrecursorVisitor<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [DoubleIndexed<Precursor>],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
        offsets: Vec<usize>,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets,
        }
    }

    fn visit_spectrum_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, descr) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            *descr.source_index_mut() = val;
        }
        self.offsets = offsets
    }

    fn visit_isolation_window(&mut self, spec_arr: &StructArray, index: usize) {
        let root = spec_arr.column(index).as_struct();
        if let Some(arr) = root.column_by_name("target") {
            let arr: &Float32Array = arr.as_primitive();
            for (offset, descr) in self.iter_instances() {
                if arr.is_null(offset) {
                    continue;
                }
                let descr = descr.description_mut();
                descr.isolation_window.target = arr.value(offset) as f32;
                descr.isolation_window.flags = mzdata::spectrum::IsolationWindowState::Explicit;
            }
        }
        if let Some(arr) = root.column_by_name("lower_bound") {
            let arr: &Float32Array = arr.as_primitive();
            for (offset, descr) in self.iter_instances() {
                if arr.is_null(offset) {
                    continue;
                }
                descr.description_mut().isolation_window.lower_bound = arr.value(offset) as f32;
                descr.description_mut().isolation_window.flags =
                    mzdata::spectrum::IsolationWindowState::Explicit;
            }
        }
        if let Some(arr) = root.column_by_name("upper_bound") {
            let arr: &Float32Array = arr.as_primitive();
            for (offset, descr) in self.iter_instances() {
                if arr.is_null(offset) {
                    continue;
                }
                descr.description_mut().isolation_window.upper_bound = arr.value(offset) as f32;
                descr.description_mut().isolation_window.flags =
                    mzdata::spectrum::IsolationWindowState::Explicit;
            }
        }
    }

    fn visit_activation(&mut self, spec_arr: &StructArray, index: usize) {
        let spec_arr = spec_arr.column(index).as_struct();
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();

        for (i, descr) in self.iter_instances() {
            let params = params_array.value(i);
            let params = params.as_struct();

            let params = ParameterVisitor::new(params).build();
            let descr = descr.description_mut();
            for p in params {
                if let Some(acc) = p.curie() {
                    match acc {
                        mzdata::params::CURIE {
                            controlled_vocabulary: mzdata::params::ControlledVocabulary::MS,
                            accession: 1000045,
                        } => {
                            let val: mzdata::params::Value = p.value.into();
                            descr.activation.energy = val.to_f32().unwrap();
                        }
                        _ => {
                            let p: mzdata::Param = p.into();
                            if mzdata::spectrum::Activation::is_param_activation(&p) {
                                descr.activation.methods_mut().push(p.into());
                            } else {
                                descr.activation.add_param(p);
                            }
                        }
                    }
                } else {
                    descr.activation.add_param(p.into());
                }
            }
        }
    }

    fn visit_precursor_id(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        let arr = arr.as_string::<i64>();
        for (index, descr) in self.iter_instances() {
            if arr.is_null(index) {
                continue;
            }
            descr.description_mut().precursor_id = Some(arr.value(index).to_string());
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        let names = spec_arr.column_names();
        let mut visited = vec![false; spec_arr.num_columns()];
        // Must visit the index first, to infer null spacing
        if let Some(i) = names
            .iter()
            .position(|v| *v == "spectrum_index" || *v == "source_index")
        {
            self.visit_spectrum_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Precursor arrays did not contain \"spectrum_index\" column");
            panic!("Precursor arrays did not contain \"spectrum_index\" column: {names:?}");
        }

        if let Some(i) = names.iter().position(|v| *v == "precursor_index") {
            self.visit_precursor_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Precursor arrays did not contain \"precursor_index\" column");
        }

        for _col in self.metadata_map.iter() {}

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            log::trace!("Visiting precursor {colname} ({index})");
            match colname {
                "activation" => {
                    self.visit_activation(spec_arr, index);
                }
                "isolation_window" => {
                    self.visit_isolation_window(spec_arr, index);
                }
                "precursor_id" => {
                    self.visit_precursor_id(spec_arr, index);
                }
                _ => {
                    // self.visit_as_param(spec_arr, index, None, Some(colname));
                }
            }
        }
    }
}

pub(crate) struct MzSelectedIonVisitor<'a> {
    pub(crate) descriptions: &'a mut [(u64, Option<u64>, SelectedIon)],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> VisitorBuilderBase<'a, (u64, Option<u64>, SelectedIon)> for MzSelectedIonVisitor<'a> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, (u64, Option<u64>, SelectedIon)> {
        OffsetCollection::new(self.descriptions, &self.offsets)
    }

    fn metadata_map(&self) -> &'a [MetadataColumn] {
        self.metadata_map
    }
}

impl<'a> VisitorBuilder3<'a, SelectedIon> for MzSelectedIonVisitor<'a> {}

impl<'a> MzSelectedIonVisitor<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [DoubleIndexed<SelectedIon>],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
        offsets: Vec<usize>,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets,
        }
    }

    fn visit_spectrum_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, descr) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            *descr.source_index_mut() = val;
        }
        self.offsets = offsets
    }

    fn visit_selected_ion_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self.iter_instances() {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let descr = descr.description_mut();
                    descr.mz = $arr.value(i) as f64;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_peak_intensity(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let descr = descr.description_mut();
                    descr.intensity = $arr.value(i) as f32;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_charge(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, descr) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    let descr = descr.description_mut();
                    if $arr.is_null(i) {
                        descr.charge = None;
                    } else {
                        descr.charge = Some($arr.value(i) as i32);
                    }
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_ion_mobility(
        &mut self,
        spec_arr: &StructArray,
        ion_mobility_value_index: usize,
        ion_mobility_type_index: usize,
    ) {
        let arr = spec_arr.column(ion_mobility_value_index);
        if arr.null_count() == arr.len() {
            return;
        }

        let ion_mobility_types = spec_arr.column(ion_mobility_type_index);
        let ion_mobility_types = AnyCURIEArray::try_from(ion_mobility_types).unwrap();

        macro_rules! pack {
            ($arr:ident) => {
                        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if $arr.is_null(i) {
                continue;
            };
            let descr = descr.description_mut();
            let im_val = $arr.value(i);
            let im_tp = ion_mobility_types.value(i).unwrap();
            match im_tp {
                // ion mobility drift time
                curie!(MS:1002476) => descr.add_param(
                    mzdata::Param::builder()
                        .name("ion mobility drift time")
                        .curie(mzdata::curie!(MS:1002476))
                        .value(im_val)
                        .unit(Unit::Millisecond)
                        .build(),
                ),
                // inverse reduced ion mobility drift time
                curie!(MS:1002815) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("inverse reduced ion mobility drift time")
                            .curie(mzdata::curie!(MS:1002815))
                            .value(im_val)
                            .unit(Unit::VoltSecondPerSquareCentimeter)
                            .build()
                    )
                }
                // FAIMS compensation voltage
                curie!(MS:1001581) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("FAIMS compensation voltage")
                            .curie(mzdata::curie!(MS:1001581))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                // SELEXION compensation voltage
                curie!(MS:1003371) => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("SELEXION compensation voltage")
                            .curie(mzdata::curie!(MS:1003371))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                _ => todo!("{im_tp} not supported yet"),
            }
        }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            todo!("{:?} not supported for ion mobility", arr.data_type());
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        let names = spec_arr.column_names();
        let mut visited = vec![false; spec_arr.num_columns()];

        // Must visit the index first, to infer null spacing
        if let Some(i) = names
            .iter()
            .position(|v| *v == "spectrum_index" || *v == "source_index")
        {
            self.visit_spectrum_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Precursor arrays did not contain \"spectrum_index\" column");
            panic!("Precursor arrays did not contain \"spectrum_index\" column: {names:?}");
        }

        if let Some(i) = names.iter().position(|v| *v == "precursor_index") {
            self.visit_precursor_index(spec_arr, i);
            visited[i] = true;
        } else {
            log::error!("Precursor arrays did not contain \"precursor_index\" column");
        }

        for col in self.metadata_map.iter() {
            log::trace!("Visiting selected ion {col:?}");
            if let Some(accession) = col.accession {
                match accession {
                    curie!(MS:1000744) => {
                        self.visit_selected_ion_mz(spec_arr, col.index);
                        visited[col.index];
                    }
                    curie!(MS:1000041) => {
                        self.visit_charge(spec_arr, col.index);
                        visited[col.index];
                    }
                    curie!(MS:1000042) => {
                        self.visit_peak_intensity(spec_arr, col.index);
                        visited[col.index];
                    }
                    _ => {
                        self.visit_as_param(spec_arr, col.index, Some(col), None);
                        visited[col.index] = true;
                    }
                }
            }
        }

        let mut ion_mobility_value_index = None;
        let mut ion_mobility_type_index = None;

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            log::trace!("Visiting selected ion {colname} ({index})");
            match colname {
                "parameters" => {
                    self.visit_parameters(spec_arr);
                }
                "ion_mobility_value" => {
                    ion_mobility_value_index = Some(index);
                }
                "ion_mobility_type" => {
                    ion_mobility_type_index = Some(index);
                }
                "selected_ion_mz" => {
                    self.visit_selected_ion_mz(spec_arr, index);
                }
                "charge_state" => {
                    self.visit_charge(spec_arr, index);
                }
                "intensity" => {
                    self.visit_peak_intensity(spec_arr, index);
                }
                _ => {
                    self.visit_as_param(spec_arr, index, None, Some(colname));
                }
            }
        }

        match (ion_mobility_value_index, ion_mobility_type_index) {
            (Some(ion_mobility_value_index), Some(ion_mobility_type_index)) => {
                self.visit_ion_mobility(
                    spec_arr,
                    ion_mobility_value_index,
                    ion_mobility_type_index,
                );
            }
            (_, _) => {}
        }
    }
}

pub(crate) struct MzChromatogramBuilder<'a> {
    pub(crate) descriptions: &'a mut [ChromatogramDescription],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> VisitorBuilderBase<'a, ChromatogramDescription> for MzChromatogramBuilder<'a> {
    fn iter_instances(&mut self) -> OffsetCollection<'_, ChromatogramDescription> {
        OffsetCollection::new(self.descriptions, &self.offsets)
    }

    fn metadata_map(&self) -> &'a [MetadataColumn] {
        self.metadata_map
    }
}

impl<'a> VisitorBuilder1<'a, ChromatogramDescription> for MzChromatogramBuilder<'a> {}

impl<'a> MzChromatogramBuilder<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [ChromatogramDescription],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets: Vec::new(),
        }
    }

    fn visit_index(&mut self, chrom_arr: &StructArray, index: usize) {
        let arr = chrom_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, descr) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            descr.index = val as usize;
        }
        self.offsets = offsets
    }

    fn visit_id(&mut self, chrom_arr: &StructArray, index: usize) {
        let arr: &LargeStringArray = chrom_arr.column(index).as_string();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let val = arr.value(i);
            descr.id = val.to_string();
        }
    }

    fn visit_polarity(&mut self, chrom_arr: &StructArray, index: usize) {
        let polarity_arr: &Int8Array = chrom_arr.column(index).as_any().downcast_ref().unwrap();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let polarity_val = polarity_arr.value(i);
            match polarity_val {
                1 => descr.polarity = ScanPolarity::Positive,
                -1 => descr.polarity = ScanPolarity::Negative,
                _ => {
                    descr.polarity = ScanPolarity::Unknown;
                }
            }
        }
    }

    fn visit_chromatogram_type(&mut self, spec_arr: &StructArray, index: usize) {
        let spec_type_array = AnyCURIEArray::try_from(spec_arr.column(index)).unwrap();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let chromatogram_type_curie = spec_type_array.value(i).unwrap();
            let chromatogram_type =
                ChromatogramType::from_accession(chromatogram_type_curie.accession);
            if let Some(chromatogram_type) = chromatogram_type {
                descr.chromatogram_type = chromatogram_type;
            }
        }
    }

    pub(crate) fn visit(&mut self, chrom_arr: &StructArray) -> usize {
        // Must visit the index first, to infer null spacing
        self.visit_index(chrom_arr, 0);
        self.visit_id(chrom_arr, 1);

        let names = chrom_arr.column_names();
        let mut visited = vec![false; chrom_arr.num_columns()];
        visited[0] = true;
        visited[1] = true;

        for col in self.metadata_map() {
            log::trace!("Visiting chromatogram {col:?}");
            if let Some(accession) = col.accession {
                match accession {
                    curie!(MS:1000465) => {
                        self.visit_polarity(chrom_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1000626) => {
                        // chromatogram type
                        self.visit_chromatogram_type(chrom_arr, col.index);
                        visited[col.index] = true;
                    }
                    curie!(MS:1003060) => {
                        // number of data points
                        visited[col.index] = true;
                    }
                    _ => {
                        self.visit_as_param(chrom_arr, col.index, col);
                        visited[col.index] = true;
                    }
                }
            }
        }

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            log::trace!("Visiting chromatogram {colname} ({index})");
            match colname {
                "polarity" => self.visit_polarity(chrom_arr, index),
                "chromatogram_type" => self.visit_chromatogram_type(chrom_arr, index),
                "parameters" => {
                    self.visit_parameters(chrom_arr, &[]);
                }
                _ => {}
            }
        }
        self.offsets.len()
    }
}

#[derive(Default, Debug)]
pub(crate) struct AuxiliaryArrayVisitor(Vec<DataArray>);

impl AuxiliaryArrayVisitor {
    fn visit_data(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let arr = arr.as_list::<i64>();
        for (i, item) in arr.iter().enumerate() {
            if let Some(item) = item {
                self.0[i].data = item.as_primitive::<UInt8Type>().values().to_vec();
            }
        }
    }

    fn visit_name(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let visitor = ParameterVisitor::new(arr.as_struct());
        let params = visitor.build();

        for (i, param) in params.into_iter().enumerate() {
            let accession = param.curie().unwrap();
            let val = ArrayType::from_accession(accession).unwrap();
            if matches!(val, ArrayType::NonStandardDataArray { name: _ }) {
                self.0[i].name = ArrayType::nonstandard(param.value.to_string());
            } else {
                self.0[i].name = val;
            }
        }
    }

    fn visit_data_type(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let visitor = AnyCURIEArray::try_from(arr).unwrap();

        for (i, da) in self.0.iter_mut().enumerate() {
            let val = visitor.value(i).unwrap();
            da.dtype = BinaryDataArrayType::from_accession(val.into()).unwrap();
        }
    }

    fn visit_compression(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let visitor = AnyCURIEArray::try_from(arr).unwrap();

        for (i, da) in self.0.iter_mut().enumerate() {
            let val = visitor.value(i).unwrap();
            da.compression = BinaryCompressionType::from_accession(val.into()).unwrap();
        }
    }

    fn visit_unit(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let visitor = UnitArray::from(arr);

        for (i, da) in self.0.iter_mut().enumerate() {
            let val = visitor.value(i);
            da.unit = val;
        }
    }

    fn visit_parameters(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index).as_list::<i64>();

        for (i, da) in self.0.iter_mut().enumerate() {
            if arr.is_null(i) {
                continue;
            }
            let val = arr.value(i);
            let val = val.as_struct();
            da.params_mut().extend(ParameterVisitor::new(val).build());
        }
    }

    fn visit_data_processing_ref(&mut self, index: usize, arrays: &StructArray) {
        let arr = arrays.column(index);
        let arr = arr.as_string::<i64>();
        for (i, da) in self.0.iter_mut().enumerate() {
            if arr.is_null(i) {
                continue;
            }
            let val: Box<str> = arr.value(i).into();
            da.set_data_processing_reference(Some(val));
        }
    }

    pub fn visit(mut self, arrays: &StructArray) -> Vec<DataArray> {
        let n = arrays.len();
        self.0.resize(n, Default::default());
        let column_names = arrays.column_names();
        for (index, col_name) in column_names.iter().enumerate() {
            match *col_name {
                "data" => {
                    self.visit_data(index, arrays);
                }
                "name" => {
                    self.visit_name(index, arrays);
                }
                "data_type" => {
                    self.visit_data_type(index, arrays);
                }
                "compression" => {
                    self.visit_compression(index, arrays);
                }
                "unit" => {
                    self.visit_unit(index, arrays);
                }
                "parameters" => {
                    self.visit_parameters(index, arrays);
                }
                "data_processing_ref" => {
                    self.visit_data_processing_ref(index, arrays);
                }
                _ => {}
            }
        }

        self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_name() {
        let (accession, _name) = parse_column_to_curie("MS_1000511_ms_level").unwrap();
        assert_eq!(accession, mzdata::curie!(MS:1000511));
    }
}
