import logging
import json
import zipfile
import zlib

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import IO, Any, Iterator

import numpy as np
import pandas as pd

import pynumpress
import pyarrow as pa

from pyarrow import compute as pc
from pyarrow import parquet as pq

from .mz_reader import _BatchIterator, MzPeakArrayDataReader, BufferFormat
from .file_index import FileIndex, DataKind, EntityType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _value_normalize(val: dict):
    for v in val.values():
        if v is not None:
            return v
    return None


def _format_curie(curie: dict):
    if curie is None:
        return None
    elif isinstance(curie, str):
        return curie
    idx = curie['cv_id']
    acc = curie['accession']
    if idx == 1:
        return f"MS:{acc}"
    elif idx == 2:
        return f"UO:{acc:07d}"
    else:
        raise NotImplementedError()


def _format_param(param: dict):
    param = param.copy()
    param["value"] = _value_normalize(param["value"])
    param['accession'] = _format_curie(param['accession'])
    if param.get('unit'):
        param["unit"] = _format_curie(param["unit"])
    return param


def _clean_frame(df: pd.DataFrame):
    columns = df.columns[~df.isna().all(axis=0)]
    df = df[columns]
    return df


def _format_curie_arrow(arr):
    cvs = arr.field(0)
    cvs = pc.case_when(
        pc.make_struct(
            pc.equal(cvs, 1),
            pc.equal(cvs, 2)
        ),
        "MS",
        "UO",
    )
    accs = arr.field(1)
    accs = pc.utf8_lpad(
        pc.cast(
            accs,
            pa.string()
        ),
        7,
        "0"
    )
    return pc.binary_join_element_wise(cvs, accs, ":")


def _format_curies_batch(bat: pa.RecordBatch) -> pa.RecordBatch:
    for i, col in enumerate(bat.schema):
        if isinstance(col.type, pa.StructType) and col.type.names == [
            "cv_id",
            "accession",
        ]:
            c = _format_curie_arrow(bat.column(i))
            bat = bat.set_column(
                i, pa.field(col.name, c.type, col.nullable, col.metadata), c
            )
    return bat


class _AuxiliaryArrayDecoder:
    """
    A helper class for decoding extra arrays packed in with the metadata table.
    """
    compression = {
        "MS:1000576": lambda x: x,
        "MS:1000574": zlib.decompress,
        'MS:1002314': pynumpress.decode_slof,
        'MS:1002313': pynumpress.decode_pic,
        'MS:1002312': pynumpress.decode_linear,
    }

    dtypes = {
        "MS:1000519": np.int32,
        'MS:1000521': np.float32,
        "MS:1000522": np.int64,
        "MS:1000523": np.float64,
    }
    ascii_code = "MS:1001479"

    @classmethod
    def decode(cls, arr: dict):
        data: np.ndarray = arr['data']
        compression_acc: str = _format_curie(arr['compression'])
        dtype_acc: str = _format_curie(arr['data_type'])
        name_param = _format_param(arr['name'])
        if name_param['name'] == "non-standard data array":
            name = name_param['value']
        else:
            name = name_param["name"]
        parameters = [_format_param(v) for v in arr.get('parameters', [])]
        data: np.ndarray = cls.compression[compression_acc](data)
        if cls.ascii_code != dtype_acc:
            data = data.view(cls.dtypes[dtype_acc])
        else:
            raise NotImplementedError(cls.ascii_code)
        return AuxiliaryArray(name, data, parameters)


@dataclass
class AuxiliaryArray:
    """
    An extra array that was not registered as globally as part of the data schema
    that has been decoded.

    Attributes
    ----------
    name : str
        The name of the array
    values : np.ndarray
        The decoded data associated with the array
    parameters : list[dict]
        The parameters, controlled or otherwise, not already covered by the decoded array attributes
    """
    name: str
    values: np.ndarray
    parameters: list[dict]


class MzPeakSpectrumMetadataReader:
    """
    A reader for spectrum metadata in an mzPeak file.

    Attributes
    ----------
    handle : :class:`pyarrow.parquet.ParquetFile`
        The underlying Parquet file reader
    meta : :class:`pyarrow.parquet.FileMetaData`
        The metadata segment of the underlying Parquet file
    num_spectra : int
        The number of distinct spectra in the metadata table
    spectra : :class:`pandas.DataFrame`
        A data frame holding spectrum-level metadata like MS level, scan time, centroid status,
        and polarity.
    id_index : :class:`pandas.Series`
        A series mapping spectrum ID to index
    precursors : :class:`pandas.DataFrame`
        A data frame holding precursor-level metadata like precursor scan ID, isolation window,
        and activation parameters. See :attr:`MzPeakFile.selected_ions` for ion-level information.
    selected_ions : :class:`pandas.Dataframe`
        A data frame holding selected ions connected to precursors and spectra including selected
        ion m/z, charge, intensity, and possibly ion mobility.
    scans : :class:`pandas.Dataframe`
        A data frame holding scan-level metadata like scan start time, injection time, filter strings
        and scan ranges.
    """
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    num_spectra: int
    num_spectrum_points: int

    spectrum_index_i: int
    scan_index_i: int
    precursor_index_i: int
    selected_ion_i: int

    id_index: pd.Series
    spectra: pd.DataFrame
    scans: pd.DataFrame
    precursors: pd.DataFrame
    selected_ions: pd.DataFrame

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = handle.metadata
        self.num_spectra = int(handle.metadata.metadata[b"spectrum_count"])
        self.num_spectrum_points = int(
            handle.metadata.metadata[b"spectrum_data_point_count"]
        )
        self._infer_schema_idx()
        self._read_spectra()
        self._read_scans()
        self._read_precursors()
        self._read_selected_ions()

    def tic(self):
        return np.array(self.spectra["time"]), np.array(
            self.spectra["total_ion_current"]
        )

    def bpc(self):
        return np.array(self.spectra["time"]), np.array(
            self.spectra["base_peak_intensity"]
        )

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == "spectrum.index":
                self.spectrum_index_i = i
            elif col.path_in_schema == "scan.spectrum_index":
                self.scan_index_i = i
            elif col.path_in_schema == "precursor.spectrum_index":
                self.precursor_index_i = i
            elif col.path_in_schema == "selected_ion.spectrum_index":
                self.selected_ion_i = i

    def _read_spectra(self):
        spectra = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.spectrum_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["spectrum"])
                spectra.append(bat.filter(bat[0].is_valid()))

        if not spectra:
            self.spectra = pd.DataFrame([], columns=['index', 'id', ])
        else:
            bat = pa.record_batch(pa.concat_tables(spectra)["spectrum"].combine_chunks())
            bat = _format_curies_batch(bat)
            self.spectra = _clean_frame(
                bat
                .to_pandas()
                .set_index("index")
            )
        self.id_index = self.spectra[["id"]].reset_index().set_index("id")["index"]

    def _read_scans(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.scan_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["scan"])
                blocks.append(bat.filter(bat[0].is_valid()))

        if blocks:
            bat = pa.record_batch(pa.concat_tables(blocks)["scan"].combine_chunks())
            bat = _format_curies_batch(bat)
            self.scans = _clean_frame(
                bat
                .to_pandas()
                .set_index("spectrum_index")
            )
        else:
            self.scans = pd.DataFrame([], columns=['spectrum_index', ])

    def _read_precursors(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.precursor_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["precursor"])
                blocks.append(bat.filter(bat[0].is_valid()))
        if blocks:
            bat = pa.record_batch(pa.concat_tables(blocks)["precursor"].combine_chunks())
            bat = _format_curies_batch(bat)
            self.precursors = _clean_frame(
                bat
                .to_pandas()
                .set_index("spectrum_index")
            )
        else:
            self.precursors = pd.DataFrame([], columns=['spectrum_index', 'precursor_index', ])

    def _read_selected_ions(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.selected_ion_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["selected_ion"])
                blocks.append(bat.filter(bat[0].is_valid()))

        if blocks:
            bat = pa.record_batch(pa.concat_tables(blocks)["selected_ion"].combine_chunks())
            bat = _format_curies_batch(bat)
            self.selected_ions = _clean_frame(
                bat
                .to_pandas()
                .set_index("spectrum_index")
            )
        else:
            self.selected_ions = pd.DataFrame([], columns=['spectrum_index', 'precursor_index', ])

    def __getitem__(self, i: int | str):
        if isinstance(i, str):
            i = self.id_index[i]
        spec = self.spectra.loc[i].to_dict()
        spec["parameters"] = [_format_param(v) for v in spec["parameters"]]
        spec["scans"] = self.scans.loc[i].to_dict()
        if isinstance(spec['scans'], dict):
            spec["scans"]["parameters"] = [
                _format_param(v) for v in spec["scans"]["parameters"]
            ]
            spec['scans'] = [spec['scans']]
        else:
            for scan in spec['scans']:
                scan["parameters"] = [_format_param(v) for v in scan["parameters"]]
        try:
            precursors_of = self.precursors.loc[[i]]
            precursors_of["activation"] = precursors_of["activation"].apply(
                lambda x: [_format_param(v) for v in x["parameters"]]
            )
            try:
                ions = self.selected_ions.loc[[i]]
                ions["parameters"] = ions["parameters"].apply(
                    lambda x: [_format_param(v) for v in x]
                )
                precursors_of = precursors_of.merge(ions, on="precursor_index")
            except KeyError:
                pass
            spec["precursors"] = precursors_of.to_dict("records")
        except KeyError:
            pass
        spec["index"] = i
        if 'auxiliary_arrays' in spec:
            for v in spec.pop("auxiliary_arrays"):
                v = _AuxiliaryArrayDecoder.decode(v)
                spec[v.name] = v.values
        return spec

    def __len__(self):
        return self.spectra.index.size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.handle})"

    def _get_mz_delta_model(self):
        if "median_delta" in self.spectra:
            return self.spectra["median_delta"].to_numpy()
        elif "mz_delta_model" in self.spectra:
            return self.spectra["mz_delta_model"].to_numpy()
        return None


class MzPeakChromatogramMetadataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    num_chromatograms: int
    num_chromatogram_points: int

    chromatogram_index_i: int
    precursor_index_i: int
    selected_ion_i: int

    id_index: pd.Series
    chromatograms: pd.DataFrame
    precursors: pd.DataFrame
    selected_ions: pd.DataFrame

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = handle.metadata
        self.num_chromatograms = int(handle.metadata.metadata[b"chromatogram_count"])
        self.num_chromatogram_points = int(
            handle.metadata.metadata[b"chromatogram_data_point_count"]
        )
        self._infer_schema_idx()
        self._read_chromatograms()
        self._read_precursors()
        self._read_selected_ions()

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == "chromatogram.index":
                self.chromatogram_index_i = i
            elif col.path_in_schema == "precursor.spectrum_index":
                self.precursor_index_i = i
            elif col.path_in_schema == "selected_ion.spectrum_index":
                self.selected_ion_i = i

    def _read_chromatograms(self):
        chromatograms = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.chromatogram_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["chromatogram"])
                chromatograms.append(bat.filter(bat[0].is_valid()))

        if not chromatograms:
            self.chromatograms = pd.DataFrame(
                [],
                columns=[
                    "index",
                    "id",
                ],
            )
        else:
            bat = pa.record_batch(
                pa.concat_tables(chromatograms)["chromatogram"].combine_chunks()
            )
            bat = _format_curies_batch(bat)
            self.chromatograms = _clean_frame(bat.to_pandas().set_index("index"))
        self.id_index = self.chromatograms[["id"]].reset_index().set_index("id")["index"]

    def _read_precursors(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.precursor_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["precursor"])
                blocks.append(bat.filter(bat[0].is_valid()))
        if blocks:
            bat = pa.record_batch(
                pa.concat_tables(blocks)["precursor"].combine_chunks()
            )
            bat = _format_curies_batch(bat)
            self.precursors = _clean_frame(bat.to_pandas().set_index("spectrum_index"))
        else:
            self.precursors = pd.DataFrame(
                [],
                columns=[
                    "spectrum_index",
                    "precursor_index",
                ],
            )

    def _read_selected_ions(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.selected_ion_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["selected_ion"])
                blocks.append(bat.filter(bat[0].is_valid()))

        if blocks:
            bat = pa.record_batch(
                pa.concat_tables(blocks)["selected_ion"].combine_chunks()
            )
            bat = _format_curies_batch(bat)
            self.selected_ions = _clean_frame(
                bat.to_pandas().set_index("spectrum_index")
            )
        else:
            self.selected_ions = pd.DataFrame(
                [],
                columns=[
                    "spectrum_index",
                    "precursor_index",
                ],
            )

    def __getitem__(self, i: int | str):
        if isinstance(i, str):
            i = self.id_index[i]
        spec = self.chromatograms.loc[i].to_dict()
        spec["parameters"] = [_format_param(v) for v in spec["parameters"]]
        try:
            precursors_of = self.precursors.loc[[i]]
            precursors_of["activation"] = precursors_of["activation"].apply(
                lambda x: [_format_param(v) for v in x["parameters"]]
            )
            try:
                ions = self.selected_ions.loc[[i]]
                ions["parameters"] = ions["parameters"].apply(
                    lambda x: [_format_param(v) for v in x]
                )
                precursors_of = precursors_of.merge(ions, on="precursor_index")
            except KeyError:
                pass
            spec["precursors"] = precursors_of.to_dict("records")
        except KeyError:
            pass
        spec["index"] = i
        if "auxiliary_arrays" in spec:
            for v in spec.pop("auxiliary_arrays"):
                v = _AuxiliaryArrayDecoder.decode(v)
                spec[v.name] = v.values
        return spec


_SpectrumType = dict[str, Any]


class MzPeakFileIter(Iterator[_SpectrumType]):
    archive: "MzPeakFile"
    index: int
    buffer_format: BufferFormat
    data_iter: _BatchIterator
    peeked: tuple[int, pa.StructArray] | None

    def __init__(self, archive: "MzPeakFile"):
        self.archive = archive
        self.index = 0
        self.buffer_format = archive.spectrum_data.buffer_format()
        self.data_iter = None
        self.peeked = None
        self._make_data_iter()

    def _make_data_iter(self):
        if self.buffer_format == BufferFormat.Point:
            it = self.archive.spectrum_data.handle.iter_batches(columns=["point"])
            self.data_iter = _BatchIterator(it, self.index)
        elif self.buffer_format == BufferFormat.Chunk:
            it = self.archive.spectrum_data.handle.iter_batches(128, columns=['chunk'])
            self.data_iter = _BatchIterator(it, self.index)
        else:
            raise ValueError(self.buffer_format)

    def _format_data_buffer(self, index: int, buffers: pa.StructArray):
        if self.buffer_format == BufferFormat.Point:
            return self.archive.spectrum_data._clean_point_batch(
                buffers,
                self.archive.spectrum_data._delta_model_series[index] if self.archive.spectrum_data._delta_model_series is not None else None
            )
        elif self.buffer_format == BufferFormat.Chunk:
            return self.archive.spectrum_data._expand_chunks(
                buffers,
                delta_model=self.archive.spectrum_data._delta_model_series[index]
                if self.archive.spectrum_data._delta_model_series is not None
                else None,
            )
        else:
            raise ValueError(self.buffer_format)

    def __iter__(self):
        return self

    def seek(self, index: int):
        self.index = index
        self.data_iter.seek(index)

    def __next__(self):
        meta = self.archive.spectrum_metadata[self.index]
        if self.peeked:
            if self.peeked[0] == self.index:
                index, data = self.peeked
                self.peeked = None
            else:
                index, data = next(self.data_iter)
        else:
            index, data = next(self.data_iter)
        if index == self.index:
            data = self._format_data_buffer(index, data)
            meta.update(data)
        else:
            self.peeked = (index, data)
        self.index += 1
        return meta

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index}, {self.archive})"


class MzPeakFile(Sequence[_SpectrumType]):
    """
    An mzPeak reader for mass spectra, chromatograms, and other
    data types.

    This may be initialized from a path to a packed zip archive
    or an unpacked directory.

    This type is an :class:`Iterable` over spectra with support
    for point and slicing access.

    Attributes
    ----------
    spectrum_data : :class:`~.MzPeakArrayDataReader`
        The facet of the data file for reading spectrum signal data from. This
        may be profile or centroid data, depending upon what was stored in the
        file.
    spectrum_metadata : :class:`~.MzPeakSpectrumMetadataReader`
        The facet of the data file for reading spectrum descriptive metadata,
        like scan time, MS level, precursor information, et cetera. Should not
        be necessary to interact with this attribute directly. Instead, see
        :attr:`MzPeakFile.spectra`, :attr:`MzPeakFile.precursors`, :attr:`MzPeakFile.scans`
        and :attr:`MzPeakFile.selected_ions`.
    spectrum_peak_data : :class:`~.MzPeakArrayDataReader` or :const:`None`
        The facet of the data file for reading explicitly stored spectrum centroid
        data from. This will only be present if the file was written with a separate
        centroid stream to store both centroids and profile data side-by-side, as
        in some instrument vendor formats.
    chromatogram_data : :class:`~.MzPeakArrayDataReader` or :const:`None`
        The facet of the data file for reading chromatogram signal data from. This
        will only be present if the writer specifically writes chromatogram data.
    chromatogram_metadata : :class:`~.MzPeakChromatogramMetadataReader`
        The facet of the data file for reading chromatogram descriptive metadata.
        Should not be necessary to interact with this attribute directly. Instead, see
        :attr:`MzPeakFile.chromatograms`
    file_index : :class:`~.FileIndex`
        A listing of the recorded files within the archive, mapping names to specific
        data content types.
    file_metadata: dict[str, Any]
        TODO: document this
    spectra : :class:`pandas.DataFrame`
        A data frame holding spectrum-level metadata like MS level, scan time, centroid status,
        and polarity.
    precursors : :class:`pandas.DataFrame`
        A data frame holding precursor-level metadata like precursor scan ID, isolation window,
        and activation parameters. See :attr:`MzPeakFile.selected_ions` for ion-level information.
    selected_ions : :class:`pandas.Dataframe`
        A data frame holding selected ions connected to precursors and spectra including selected
        ion m/z, charge, intensity, and possibly ion mobility.
    scans : :class:`pandas.Dataframe`
        A data frame holding scan-level metadata like scan start time, injection time, filter strings
        and scan ranges.
    chromatograms : :class:`pandas.DataFrame` or :const:`None`
        A data frame holding chromatogram-level metadata. This will only be present if
        :attr:`chromatogram_metadata` is present.
    """
    _archive: zipfile.ZipFile | Path

    spectrum_data: MzPeakArrayDataReader | None = None
    spectrum_metadata: MzPeakSpectrumMetadataReader | None = None
    spectrum_peak_data: MzPeakArrayDataReader | None = None

    chromatogram_metadata: MzPeakChromatogramMetadataReader | None = None
    chromatogram_data: MzPeakArrayDataReader | None = None

    file_metadata: dict[str, Any]

    file_index: FileIndex

    @property
    def filename(self) -> str | None:
        '''The name of the data file'''
        if isinstance(self._archive, Path):
            return self._archive.name
        elif isinstance(self._archive, zipfile.ZipFile):
            return self._archive.filename

    def _from_directory(self, path: Path):
        self._archive = path
        index_path = path / FileIndex.FILE_NAME
        if index_path.exists():
            self.file_index = FileIndex.from_json(json.load(index_path.open()))
            for e in self.file_index:
                f = path / e.name
                match e.entry_type():
                    case (EntityType.Spectrum, DataKind.DataArrays):
                        self.spectrum_data = MzPeakArrayDataReader(
                            pa.OSFile(f),
                            namespace="spectrum",
                        )
                    case (EntityType.Spectrum, DataKind.Metadata):
                        self.spectrum_metadata = MzPeakSpectrumMetadataReader(
                            pa.OSFile(f),
                        )
                    case (EntityType.Spectrum, DataKind.Peaks):
                        self.spectrum_peak_data = MzPeakArrayDataReader(
                            pa.OSFile(f),
                            namespace="spectrum",
                        )
                    case (EntityType.Chromatogram, DataKind.DataArrays):
                        self.chromatogram_data = MzPeakArrayDataReader(
                            pa.OSFile(f),
                            namespace="chromatogram",
                        )
                    case (EntityType.Chromatogram, DataKind.Metadata):
                        self.chromatogram_metadata = MzPeakChromatogramMetadataReader(
                            pa.OSFile(f)
                        )
                    case _:
                        pass

        for f in path.glob("*mzpeak"):
            if not f.is_file():
                continue
            if f.name.endswith("spectra_data.mzpeak") and not self.spectrum_data:
                self.spectrum_data = MzPeakArrayDataReader(
                    pa.OSFile(f),
                    namespace="spectrum",
                )
            elif f.name.endswith("spectra_metadata.mzpeak") and not self.spectrum_metadata:
                self.spectrum_metadata = MzPeakSpectrumMetadataReader(
                    pa.OSFile(f),
                )
            elif f.name.endswith("spectra_peaks.mzpeak") and not self.spectrum_peak_data:
                self.spectrum_peak_data = MzPeakArrayDataReader(
                    pa.OSFile(f),
                    namespace="spectrum",
                )
            elif (
                f.name.endswith("chromatograms_metadata.mzpeak")
                and not self.chromatogram_metadata
            ):
                self.chromatogram_metadata = MzPeakChromatogramMetadataReader(
                    pa.OSFile(f)
                )
            elif (
                f.name.endswith("chromatograms_data.mzpeak") and not self.chromatogram_data
            ):
                self.chromatogram_data = MzPeakArrayDataReader(
                    pa.OSFile(f),
                    namespace="chromatogram",
                )

    def _from_zip_archive(self, archive: zipfile.ZipFile):
        self._archive = archive
        try:
            f = archive.getinfo(FileIndex.FILE_NAME)
            self.file_index = FileIndex.from_json(json.load(archive.open(f)))
            for e in self.file_index:
                f = archive.open(e.name)
                match e.entry_type():
                    case (EntityType.Spectrum, DataKind.DataArrays):
                        self.spectrum_data = MzPeakArrayDataReader(
                            pa.PythonFile(f),
                            namespace="spectrum",
                        )
                    case (EntityType.Spectrum, DataKind.Metadata):
                        self.spectrum_metadata = MzPeakSpectrumMetadataReader(
                            pa.PythonFile(f),
                        )
                    case (EntityType.Spectrum, DataKind.Peaks):
                        self.spectrum_peak_data = MzPeakArrayDataReader(
                            pa.PythonFile(f),
                            namespace="spectrum",
                        )
                    case (EntityType.Chromatogram, DataKind.DataArrays):
                        self.chromatogram_data = MzPeakArrayDataReader(
                            pa.PythonFile(f),
                            namespace="chromatogram",
                        )
                    case (EntityType.Chromatogram, DataKind.Metadata):
                        self.chromatogram_metadata = MzPeakChromatogramMetadataReader(
                            pa.PythonFile(f)
                        )
                    case _:
                        pass
        except KeyError:
            pass
        for f in archive.filelist:
            if f.filename.endswith("spectra_data.mzpeak") and not self.spectrum_data:
                self.spectrum_data = MzPeakArrayDataReader(
                    pa.PythonFile(archive.open(f)),
                    namespace="spectrum",
                )
            elif (
                f.filename.endswith("spectra_metadata.mzpeak")
                and not self.spectrum_metadata
            ):
                self.spectrum_metadata = MzPeakSpectrumMetadataReader(
                    pa.PythonFile(archive.open(f)),
                )
            elif f.filename.endswith("spectra_peaks.mzpeak") and not self.spectrum_peak_data:
                self.spectrum_peak_data = MzPeakArrayDataReader(
                    pa.PythonFile(archive.open(f)),
                    namespace="spectrum",
                )
            elif (
                f.filename.endswith("chromatograms_metadata.mzpeak")
                and not self.chromatogram_metadata
            ):
                self.chromatogram_metadata = MzPeakChromatogramMetadataReader(
                    pa.PythonFile(archive.open(f))
                )
            elif (
                f.filename.endswith("chromatograms_data.mzpeak")
                and not self.chromatogram_data
            ):
                self.chromatogram_data = MzPeakArrayDataReader(
                    pa.PythonFile(archive.open(f)),
                    namespace="chromatogram",
                )

    def _from_path(self, path: Path):
        if path.is_dir():
            self._from_directory(path)
        else:
            archive = zipfile.ZipFile(path)
            self._from_zip_archive(archive)

    def spectra_signal_for_indices(self, index_range: slice | list[int]) -> dict[str, np.ndarray]:
        return self.spectrum_data.read_data_for_range(index_range)

    def _init_metadata(self):
        metadata = {}
        if self.spectrum_metadata:
            for k, v in self.spectrum_metadata.meta.metadata.items():
                k = k.decode("utf8")
                if k == "ARROW:schema":
                    continue
                v = json.loads(v)
                metadata[k] = v
        self.file_metadata = metadata

        if self.spectrum_data and self.spectrum_metadata:
            self.spectrum_data._delta_model_series = (
                self.spectrum_metadata._get_mz_delta_model()
            )

    def __init__(self, path: str | Path | zipfile.ZipFile | IO[bytes]):
        self.file_index = FileIndex()
        if isinstance(path, zipfile.ZipFile):
            self._from_zip_archive(self._archive)
        elif isinstance(path, (str, Path)):
            self._from_path(Path(path))
        else:
            self._from_zip_archive(zipfile.ZipFile(path))

        self._init_metadata()

    def read_spectrum(
        self, index: int | str | Iterable[int | str] | slice
    ) -> _SpectrumType | list[_SpectrumType]:
        if isinstance(index, (int, str)):
            spec = self.spectrum_metadata[index]
            index = spec["index"]
            data = self.spectrum_data[index]
            spec.update(data)
        elif isinstance(index, Iterable):
            if not index:
                return []
            spec = [self.read_spectrum(i) for i in index]
        elif isinstance(index, slice):
            start = index.start or 0
            end = index.stop or len(self)
            step = index.step or 1
            spec = self.read_spectrum(range(start, end, step))
        return spec

    def read_chromatogram(
        self, index: int | str | Iterable[int | str] | slice
    ) -> _SpectrumType | list[_SpectrumType]:
        if isinstance(index, (int, str)):
            chrom = self.chromatogram_metadata[index]
            index = chrom["index"]
            data = self.chromatogram_data[index]
            chrom.update(data)
        elif isinstance(index, Iterable):
            if not index:
                return []
            chrom = [self.read_chromatogram(i) for i in index]
        elif isinstance(index, slice):
            start = index.start or 0
            end = index.stop or len(self)
            step = index.step or 1
            chrom = self.read_chromatogram(range(start, end, step))
        return chrom

    def __repr__(self):
        return f"{self.__class__.__name__}({self._archive.filename!r})"

    def __getitem__(self, index: int | str | Iterable[int | str] | slice) -> _SpectrumType | list[_SpectrumType]:
        '''An alias for :meth:`read_spectrum`.'''
        return self.read_spectrum(index)

    def __iter__(self) -> MzPeakFileIter:
        return MzPeakFileIter(self)

    def __len__(self):
        return len(self.spectrum_metadata)

    def tic(self) -> tuple[np.ndarray, np.ndarray]:
        return self.spectrum_metadata.tic()

    def bpc(self) -> tuple[np.ndarray, np.ndarray]:
        return self.spectrum_metadata.bpc()

    @property
    def has_secondary_peaks_data(self) -> bool:
        return self.spectrum_peak_data is not None

    @property
    def spectra(self) -> pd.DataFrame:
        return self.spectrum_metadata.spectra

    @property
    def precursors(self) -> pd.DataFrame:
        return self.spectrum_metadata.precursors

    @property
    def selected_ions(self) -> pd.DataFrame:
        return self.spectrum_metadata.selected_ions

    @property
    def scans(self) -> pd.DataFrame:
        return self.spectrum_metadata.scans

    @property
    def chromatograms(self) -> pd.DataFrame | None:
        if self.chromatogram_metadata is not None:
            return self.chromatogram_metadata.chromatograms
