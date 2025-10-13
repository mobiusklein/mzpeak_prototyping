use std::fs;
use std::io::{self, prelude::*};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use parquet::file::reader::{ChunkReader, Length};
use zip::{
    CompressionMethod,
    read::{Config, ZipArchive},
    result::ZipResult,
    write::{SimpleFileOptions, ZipWriter},
};

use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArchiveState {
    #[default]
    Opened,
    SpectrumDataArrays,
    SpectrumMetadata,
    ChromatogramDataArrays,
    ChromatogramMetadata,
    Other,
    Closed,
}

fn file_options() -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .large_file(true)
}

#[derive(Debug)]
pub struct ZipArchiveWriter<W: Write + Send + Seek> {
    archive_writer: ZipWriter<W>,
    state: ArchiveState,
}

impl<W: Write + Send + Seek> Write for ZipArchiveWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.archive_writer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.archive_writer.flush()
    }
}

impl<W: Write + Send + Seek> ZipArchiveWriter<W> {
    pub fn new(writer: W) -> Self {
        let archive_writer = ZipWriter::new(writer);
        let state = ArchiveState::Opened;
        Self {
            archive_writer,
            state,
        }
    }

    pub fn start_spectrum_data(&mut self) -> ZipResult<()> {
        self.archive_writer.start_file(
            MzPeakArchiveType::SpectrumDataArrays.tag_file_suffix(),
            file_options(),
        )?;
        self.state = ArchiveState::SpectrumDataArrays;
        Ok(())
    }

    pub fn start_spectrum_metadata(&mut self) -> ZipResult<()> {
        self.archive_writer.start_file(
            MzPeakArchiveType::SpectrumMetadata.tag_file_suffix(),
            file_options(),
        )?;
        self.state = ArchiveState::SpectrumMetadata;
        Ok(())
    }

    pub fn start_chromatogram_metadata(&mut self) -> ZipResult<()> {
        self.archive_writer.start_file(
            MzPeakArchiveType::ChromatogramMetadata.tag_file_suffix(),
            file_options(),
        )?;
        self.state = ArchiveState::ChromatogramMetadata;
        Ok(())
    }

    pub fn start_chromatogram_data(&mut self) -> ZipResult<()> {
        self.archive_writer.start_file(
            MzPeakArchiveType::ChromatogramDataArrays.tag_file_suffix(),
            file_options(),
        )?;
        self.state = ArchiveState::ChromatogramDataArrays;
        Ok(())
    }

    pub fn start_other<S: AsRef<str>>(&mut self, name: Option<&S>) -> ZipResult<()> {
        self.archive_writer.start_file(
            name.map(|s| s.as_ref())
                .unwrap_or_else(|| MzPeakArchiveType::Other.tag_file_suffix().as_ref()),
            file_options(),
        )?;
        self.state = ArchiveState::Other;
        Ok(())
    }

    pub fn add_other_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let path = path.as_ref();
        let p = path.file_name().map(|p| p.to_string_lossy().to_string());
        let mut handle = fs::File::open(&path)?;
        self.add_file_from_read(&mut handle, p.as_ref())
    }

    pub fn add_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        archive_type: MzPeakArchiveType,
    ) -> io::Result<()> {
        let path = path.as_ref();
        let p = path
            .file_name()
            .map(|p| p.to_string_lossy().to_string() + archive_type.tag_file_suffix());
        let mut handle = fs::File::open(&path)?;
        self.add_file_from_read(&mut handle, p.as_ref())
    }

    pub fn add_file_from_read<S: AsRef<str>>(
        &mut self,
        read: &mut impl io::Read,
        name: Option<&S>,
    ) -> io::Result<()> {
        self.start_other(name)?;

        let mut buffer = [0u8; 65536];
        loop {
            let z = read.read(&mut buffer)?;
            if z == 0 {
                break;
            }
            self.write_all(&buffer[0..z])?;
        }

        Ok(())
    }

    pub fn finish(self) -> ZipResult<W> {
        let val = self.archive_writer.finish()?;
        Ok(val)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MzPeakArchiveType {
    SpectrumMetadata,
    SpectrumDataArrays,
    SpectrumPeakDataArrays,
    ChromatogramMetadata,
    ChromatogramDataArrays,
    Other,
}

impl MzPeakArchiveType {
    pub const fn tag_file_suffix(&self) -> &'static str {
        match self {
            MzPeakArchiveType::SpectrumMetadata => "spectra_metadata.mzpeak",
            MzPeakArchiveType::SpectrumDataArrays => "spectra_data.mzpeak",
            MzPeakArchiveType::SpectrumPeakDataArrays => "spectra_peaks.mzpeak",
            MzPeakArchiveType::ChromatogramMetadata => "chromatograms_metadata.mzpeak",
            MzPeakArchiveType::ChromatogramDataArrays => "chromatograms_data.mzpeak",
            MzPeakArchiveType::Other => "",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        if name.ends_with(Self::SpectrumDataArrays.tag_file_suffix()) {
            Some(Self::SpectrumDataArrays)
        } else if name.ends_with(Self::SpectrumMetadata.tag_file_suffix()) {
            Some(Self::SpectrumMetadata)
        } else if name.ends_with(Self::ChromatogramDataArrays.tag_file_suffix()) {
            Some(Self::ChromatogramDataArrays)
        } else if name.ends_with(Self::ChromatogramMetadata.tag_file_suffix()) {
            Some(Self::ChromatogramMetadata)
        } else if name.ends_with(Self::Other.tag_file_suffix()) {
            Some(Self::Other)
        } else {
            None
        }
    }
}

pub struct ArchiveFacetReader {
    archive: fs::File,
    start_offset: u64,
    length: u64,
    at: u64,
}

impl ArchiveFacetReader {
    pub fn new(archive: fs::File, start_offset: u64, length: u64, at: u64) -> Self {
        Self {
            archive,
            start_offset,
            length,
            at,
        }
    }

    pub fn try_clone(&self) -> io::Result<Self> {
        let archive = self.archive.try_clone()?;
        Ok(Self::new(archive, self.start_offset, self.length, self.at))
    }
}

impl Read for ArchiveFacetReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.length - self.at;
        let buf = if buf.len() as u64 > remaining {
            &mut buf[0..(remaining as usize)]
        } else {
            buf
        };
        self.archive.read(buf)
    }
}

impl Seek for ArchiveFacetReader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        match pos {
            io::SeekFrom::Start(offset) => {
                self.archive
                    .seek(io::SeekFrom::Start(self.start_offset + offset))?;
                self.at = offset;
                Ok(offset)
            }
            io::SeekFrom::End(offset) => {
                if offset < 0 {
                    let point = self.start_offset + self.length;
                    let point = point.saturating_sub(offset.abs() as u64);
                    self.archive.seek(io::SeekFrom::Start(point))?;
                    self.at = self.length.saturating_sub(offset.abs() as u64);
                } else {
                    self.archive
                        .seek(io::SeekFrom::Start(self.start_offset + self.length))?;
                    self.at = self.length;
                }
                Ok(self.at)
            }
            io::SeekFrom::Current(_offset) => {
                todo!()
            }
        }
    }
}

impl Length for ArchiveFacetReader {
    fn len(&self) -> u64 {
        self.length
    }
}

impl ChunkReader for ArchiveFacetReader {
    type T = io::BufReader<ArchiveFacetReader>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        let mut handle = self.try_clone()?;
        handle.seek(io::SeekFrom::Start(start))?;

        Ok(io::BufReader::new(handle))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        // log::info!("Read {start}-{}", start + length as u64);
        let mut buffer = Vec::with_capacity(length);
        let mut reader = self.try_clone()?;
        reader.seek(io::SeekFrom::Start(start))?;
        let read = reader.take(length as _).read_to_end(&mut buffer)?;

        if read != length {
            return Err(parquet::errors::ParquetError::EOF(format!(
                "Expected to read {} bytes, read only {}",
                length, read
            )));
        }
        Ok(buffer.into())
    }
}

pub struct ZipArchiveSource {
    archive_file: fs::File,
    archive_offset: Config,
    pub file_names: Vec<String>,
}

impl ZipArchiveSource {
    pub fn new(archive_file: fs::File) -> io::Result<Self> {
        let arch = ZipArchive::new(archive_file)?;
        let offset = arch.offset();
        let file_names: Vec<String> = arch.file_names().map(|s| s.to_string()).collect();
        let archive_file = arch.into_inner();
        let archive_offset = Config {
            archive_offset: zip::read::ArchiveOffset::Known(offset),
        };
        Ok(Self {
            archive_file,
            archive_offset,
            file_names,
        })
    }

    pub fn archive(&self) -> io::Result<ZipArchive<fs::File>> {
        let handle = self.archive_file.try_clone()?;
        let archive = ZipArchive::with_config(self.archive_offset.clone(), handle)?;
        Ok(archive)
    }

    pub fn open_entry_by_index(&self, index: usize) -> io::Result<ArchiveFacetReader> {
        let handle = self.archive_file.try_clone()?;
        let mut archive = ZipArchive::with_config(self.archive_offset.clone(), handle)?;
        let handle = archive.by_index(index)?;
        match handle.compression() {
            CompressionMethod::Stored => {}
            method => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "Compression method {method:?} isn't supported. Only Stored is supported"
                    ),
                ));
            }
        }
        let start_offset = handle.data_start();
        let length = handle.size();
        Ok(ArchiveFacetReader::new(
            self.archive_file.try_clone()?,
            start_offset,
            length,
            0,
        ))
    }

    pub fn metadata_for_index(&self, index: usize) -> io::Result<ArrowReaderMetadata> {
        let handle = self.open_entry_by_index(index)?;
        let opts = ArrowReaderOptions::new().with_page_index(true);
        Ok(ArrowReaderMetadata::load(&handle, opts)?)
    }

    pub fn read_index(
        &self,
        index: usize,
        metadata: Option<ArrowReaderMetadata>,
    ) -> io::Result<ParquetRecordBatchReaderBuilder<ArchiveFacetReader>> {
        let metadata = if let Some(metadata) = metadata {
            metadata
        } else {
            self.metadata_for_index(index)?
        };

        let handle = self.open_entry_by_index(index)?;
        Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(
            handle, metadata,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct MzPeakArchiveEntry {
    pub metadata: ArrowReaderMetadata,
    pub entry_index: usize,
    pub name: String,
    pub entry_type: MzPeakArchiveType,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct SchemaMetadataManager {
    pub(crate) spectrum_data_arrays: Option<MzPeakArchiveEntry>,
    pub(crate) spectrum_metadata: Option<MzPeakArchiveEntry>,
    pub(crate) peaks_data_arrays: Option<MzPeakArchiveEntry>,
    pub(crate) chromatogram_metadata: Option<MzPeakArchiveEntry>,
    pub(crate) chromatogram_data_arrays: Option<MzPeakArchiveEntry>,
}

pub trait ArchiveSource: Sized + 'static {
    type File: ChunkReader + 'static;

    fn from_path(path: PathBuf) -> io::Result<Self>;
    fn file_names(&self) -> &[String];
    fn open_entry_by_index(&self, index: usize) -> io::Result<Self::File>;

    fn metadata_for_index(&self, index: usize) -> io::Result<ArrowReaderMetadata> {
        let handle = self.open_entry_by_index(index)?;
        let opts = ArrowReaderOptions::new().with_page_index(true);
        Ok(ArrowReaderMetadata::load(&handle, opts)?)
    }

    fn read_index(
        &self,
        index: usize,
        metadata: Option<ArrowReaderMetadata>,
    ) -> io::Result<ParquetRecordBatchReaderBuilder<Self::File>> {
        let metadata = if let Some(metadata) = metadata {
            metadata
        } else {
            self.metadata_for_index(index)?
        };

        let handle = self.open_entry_by_index(index)?;
        Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(
            handle, metadata,
        ))
    }
}

impl ArchiveSource for ZipArchiveSource {
    type File = ArchiveFacetReader;

    fn from_path(path: PathBuf) -> io::Result<Self> {
        Self::new(fs::File::open(path)?)
    }

    fn file_names(&self) -> &[String] {
        &self.file_names
    }

    fn open_entry_by_index(&self, index: usize) -> io::Result<Self::File> {
        self.open_entry_by_index(index)
    }
}

pub struct DirectorySource {
    archive_path: PathBuf,
    pub file_names: Vec<String>,
}

impl DirectorySource {
    pub fn new(archive_path: PathBuf) -> io::Result<Self> {
        let read_dir = fs::read_dir(&archive_path)?;

        let file_names = read_dir
            .flatten()
            .map(|p| p.file_name().to_string_lossy().to_string())
            .collect();

        Ok(Self {
            archive_path,
            file_names,
        })
    }

    pub fn open_entry_by_index(&self, index: usize) -> io::Result<ArchiveFacetReader> {
        let name = self.file_names.get(index).ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("file at {index} not found in directory"),
        ))?;
        let path = self.archive_path.join(name);
        let fh = fs::File::open(&path)?;
        let meta = fs::metadata(&path)?;
        let length = meta.len();
        Ok(ArchiveFacetReader::new(fh, 0, length, 0))
    }

    pub fn metadata_for_index(&self, index: usize) -> io::Result<ArrowReaderMetadata> {
        let handle = self.open_entry_by_index(index)?;
        let opts = ArrowReaderOptions::new().with_page_index(true);
        Ok(ArrowReaderMetadata::load(&handle, opts)?)
    }

    pub fn read_index(
        &self,
        index: usize,
        metadata: Option<ArrowReaderMetadata>,
    ) -> io::Result<ParquetRecordBatchReaderBuilder<ArchiveFacetReader>> {
        let metadata = if let Some(metadata) = metadata {
            metadata
        } else {
            self.metadata_for_index(index)?
        };

        let handle = self.open_entry_by_index(index)?;
        Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(
            handle, metadata,
        ))
    }
}

impl ArchiveSource for DirectorySource {
    type File = ArchiveFacetReader;

    fn file_names(&self) -> &[String] {
        &self.file_names
    }

    fn open_entry_by_index(&self, index: usize) -> io::Result<Self::File> {
        self.open_entry_by_index(index)
    }

    fn from_path(path: PathBuf) -> io::Result<Self> {
        Self::new(path)
    }
}

pub struct ArchiveReader<T: ArchiveSource + 'static> {
    archive: T,
    members: SchemaMetadataManager,
}

impl<T: ArchiveSource + 'static> ArchiveReader<T> {
    fn init_from_archive(archive: T) -> io::Result<Self> {
        let mut members = SchemaMetadataManager::default();
        for (i, name) in archive.file_names().iter().enumerate() {
            let metadata = archive.metadata_for_index(i)?;
            let tp = if name.ends_with(MzPeakArchiveType::SpectrumDataArrays.tag_file_suffix()) {
                MzPeakArchiveType::SpectrumDataArrays
            } else if name.ends_with(MzPeakArchiveType::SpectrumMetadata.tag_file_suffix()) {
                MzPeakArchiveType::SpectrumMetadata
            } else if name.ends_with(MzPeakArchiveType::SpectrumPeakDataArrays.tag_file_suffix()) {
                MzPeakArchiveType::SpectrumPeakDataArrays
            } else if name.ends_with(MzPeakArchiveType::ChromatogramMetadata.tag_file_suffix()) {
                MzPeakArchiveType::ChromatogramMetadata
            } else if name.ends_with(MzPeakArchiveType::ChromatogramDataArrays.tag_file_suffix()) {
                MzPeakArchiveType::ChromatogramDataArrays
            } else {
                MzPeakArchiveType::Other
            };
            let entry = MzPeakArchiveEntry {
                entry_index: i,
                metadata,
                name: name.clone(),
                entry_type: tp,
            };
            match tp {
                MzPeakArchiveType::SpectrumMetadata => {
                    members.spectrum_metadata = Some(entry);
                }
                MzPeakArchiveType::SpectrumDataArrays => {
                    members.spectrum_data_arrays = Some(entry);
                }
                MzPeakArchiveType::SpectrumPeakDataArrays => {
                    members.peaks_data_arrays = Some(entry)
                }
                MzPeakArchiveType::ChromatogramMetadata => {
                    members.chromatogram_metadata = Some(entry)
                }
                MzPeakArchiveType::ChromatogramDataArrays => {
                    members.chromatogram_data_arrays = Some(entry)
                }
                MzPeakArchiveType::Other => {}
            }
        }
        Ok(Self { archive, members })
    }

    pub fn from_path(archive_path: PathBuf) -> io::Result<Self> {
        let archive = T::from_path(archive_path)?;
        Self::init_from_archive(archive)
    }

    pub fn chromatograms_metadata(&self) -> io::Result<ParquetRecordBatchReaderBuilder<T::File>> {
        if let Some(meta) = self.members.chromatogram_metadata.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Chromatogram metadata entry not found",
            ))
        }
    }

    pub fn chromatograms_data(&self) -> io::Result<ParquetRecordBatchReaderBuilder<T::File>> {
        if let Some(meta) = self.members.chromatogram_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Chromatogram data entry not found",
            ))
        }
    }

    pub fn spectra_data(&self) -> io::Result<ParquetRecordBatchReaderBuilder<T::File>> {
        if let Some(meta) = self.members.spectrum_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum data entry not found",
            ))
        }
    }

    pub fn spectrum_peaks(&self) -> io::Result<ParquetRecordBatchReaderBuilder<T::File>> {
        if let Some(meta) = self.members.peaks_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum peak data entry not found",
            ))
        }
    }

    pub fn spectrum_metadata(&self) -> io::Result<ParquetRecordBatchReaderBuilder<T::File>> {
        if let Some(meta) = self.members.spectrum_metadata.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum metadata entry not found",
            ))
        }
    }
}

pub type ZipArchiveReader = ArchiveReader<ZipArchiveSource>;
pub type DirectoryArchiveReader = ArchiveReader<DirectorySource>;
pub type AnyArchiveReader = ArchiveReader<DispatchArchiveSource>;

pub enum DispatchArchiveSource {
    Zip(ZipArchiveSource),
    Directory(DirectorySource),
}

macro_rules! dispatch {
    ($d:ident, $r:ident, $e:expr) => {
        match $d {
            DispatchArchiveSource::Zip($r) => $e,
            DispatchArchiveSource::Directory($r) => $e,
        }
    };
}

impl ArchiveSource for DispatchArchiveSource {
    type File = ArchiveFacetReader;

    fn from_path(path: PathBuf) -> io::Result<Self> {
        if fs::metadata(&path)?.is_dir() {
            return Ok(Self::Directory(DirectorySource::from_path(path)?));
        } else {
            return Ok(Self::Zip(ZipArchiveSource::from_path(path)?));
        }
    }

    fn file_names(&self) -> &[String] {
        dispatch!(self, src, { src.file_names() })
    }

    fn open_entry_by_index(&self, index: usize) -> io::Result<Self::File> {
        dispatch!(self, src, { src.open_entry_by_index(index) })
    }
}

impl ArchiveReader<DispatchArchiveSource> {
    pub fn new(file: fs::File) -> io::Result<Self> {
        let archive = DispatchArchiveSource::Zip(ZipArchiveSource::new(file)?);
        Self::init_from_archive(archive)
    }
}

impl ZipArchiveReader {
    pub fn new(file: fs::File) -> io::Result<Self> {
        let archive = ZipArchiveSource::new(file)?;
        Self::init_from_archive(archive)
    }
}

impl DirectoryArchiveReader {
    pub fn new(file: PathBuf) -> io::Result<Self> {
        let archive = DirectorySource::new(file)?;
        Self::init_from_archive(archive)
    }
}

#[cfg(test)]
mod test {
    use arrow::array::AsArray;

    use super::*;

    #[test]
    fn test_base() -> io::Result<()> {
        let arch = ZipArchiveReader::new(fs::File::open("small.mzpeak")?)?;
        let handle = arch.spectrum_metadata()?;
        let reader = handle.with_limit(5).build()?;
        for batch in reader {
            let batch = batch.unwrap();
            let spec = batch.column(0).as_struct();
            assert_eq!(spec.column_by_name("index").unwrap().len(), 5);
        }
        Ok(())
    }
}
