use std::io::{self, SeekFrom};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use bytes::Bytes;

use futures::{AsyncReadExt, FutureExt};
use object_store::parse_url;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

use async_zip::{StoredZipEntry, base::read::seek::ZipFileReader};
use object_store::{ObjectMeta, ObjectStore, path::Path as ObjectPath};
use tokio::io::{AsyncBufRead, AsyncRead, AsyncSeek, ReadBuf};
use url::Url;

use crate::archive::FileIndex;

use super::sync::{MzPeakArchiveEntry, MzPeakArchiveType, SchemaMetadataManager};

pub trait AsyncArchiveSource: Clone + 'static {
    type File: parquet::arrow::async_reader::AsyncFileReader + Unpin + Send;

    fn from_store_path(
        handle: Arc<dyn ObjectStore>,
        path: ObjectPath,
    ) -> impl Future<Output = io::Result<Self>>;
    fn file_names(&self) -> &[String];
    fn open_entry_by_index(&self, index: usize) -> impl Future<Output = io::Result<Self::File>>;
    fn open_stream(&self, name: &str) -> impl Future<Output = io::Result<Self::File>> {
        async move {
            if let Some(index) = self.file_names().iter().position(|v| v == name) {
                self.open_entry_by_index(index).await
            } else {
                Err(
                    io::Error::new(io::ErrorKind::NotFound, format!("Could not find an entry by name for \"{name}\""))
                )
            }
        }
    }
    fn metadata_for_index(
        &self,
        index: usize,
    ) -> impl Future<Output = io::Result<ArrowReaderMetadata>> {
        async move {
            let mut handle = self.open_entry_by_index(index).await?;
            let opts = ArrowReaderOptions::new().with_page_index(true);
            let meta = ArrowReaderMetadata::load_async(&mut handle, opts).await?;
            Ok(meta)
        }
    }
    fn read_index(
        &self,
        index: usize,
        metadata: Option<ArrowReaderMetadata>,
    ) -> impl Future<Output = io::Result<ParquetRecordBatchStreamBuilder<Self::File>>> {
        async move {
            let metadata = if let Some(metadata) = metadata {
                metadata
            } else {
                self.metadata_for_index(index).await?
            };

            let handle = self.open_entry_by_index(index).await?;
            Ok(ParquetRecordBatchStreamBuilder::new_with_metadata(
                handle, metadata,
            ))
        }
    }

    fn file_index(&self) -> &FileIndex;
}

#[derive(Clone)]
pub struct AsyncZipArchiveSource {
    handle: Arc<dyn ObjectStore>,
    root: ObjectMeta,
    file_names: Vec<String>,
    entries: Vec<StoredZipEntry>,
    file_index: FileIndex,
}

impl AsyncZipArchiveSource {
    pub async fn new(handle: Arc<dyn ObjectStore>, prefix: ObjectPath) -> io::Result<Self> {
        let root = handle.head(&prefix).await?;
        let reader = object_store::buffered::BufReader::new(handle.clone(), &root);

        let mut reader = match ZipFileReader::with_tokio(reader).await {
            Ok(reader) => reader,
            Err(e) => return Err(io::Error::other(e)),
        };

        let all_entries = reader.file().entries().to_vec();

        let mut entries = Vec::with_capacity(all_entries.len());
        let mut file_names = Vec::new();

        let mut file_index = FileIndex::default();

        for (i, entry) in all_entries.into_iter().enumerate() {
            if let Ok(name) = entry.filename().as_str() {
                if name == FileIndex::index_file_name() {
                    let mut handle = reader
                        .reader_without_entry(i)
                        .await
                        .map_err(io::Error::other)?;
                    let mut buf = String::new();
                    handle.read_to_string(&mut buf).await?;
                    file_index = serde_json::from_str(&buf)?;
                }
                let name = name.to_string();
                file_names.push(name);
                entries.push(entry);
            }
        }

        Ok(Self {
            handle,
            root,
            entries,
            file_names,
            file_index,
        })
    }

    pub async fn open_entry_by_index(&self, index: usize) -> io::Result<AsyncArchiveFacetReader> {
        let entry = self.entries.get(index).ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Could not find an entry with index {index}"),
        ))?;
        let start_offset = entry.header_offset() + entry.header_size();
        let length = entry.uncompressed_size();

        if !matches!(entry.compression(), async_zip::Compression::Stored) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "Compression method {:?} isn't supported. Only Stored is supported",
                    entry.compression()
                ),
            ));
        }

        Ok(AsyncArchiveFacetReader::new(
            self.handle.clone(),
            self.root.clone(),
            start_offset,
            length,
            0,
        ))
    }

    pub async fn metadata_for_index(&self, index: usize) -> io::Result<ArrowReaderMetadata> {
        let mut handle = self.open_entry_by_index(index).await?;
        let opts = ArrowReaderOptions::new().with_page_index(true);
        let meta = ArrowReaderMetadata::load_async(&mut handle, opts).await?;
        Ok(meta)
    }

    pub async fn read_index(
        &self,
        index: usize,
        metadata: Option<ArrowReaderMetadata>,
    ) -> io::Result<ParquetRecordBatchStreamBuilder<AsyncArchiveFacetReader>> {
        let metadata = if let Some(metadata) = metadata {
            metadata
        } else {
            self.metadata_for_index(index).await?
        };

        let handle = self.open_entry_by_index(index).await?;
        Ok(ParquetRecordBatchStreamBuilder::new_with_metadata(
            handle, metadata,
        ))
    }

    pub fn file_names(&self) -> &[String] {
        &self.file_names
    }
}

impl AsyncArchiveSource for AsyncZipArchiveSource {
    type File = AsyncArchiveFacetReader;

    fn from_store_path(
        handle: Arc<dyn ObjectStore>,
        path: ObjectPath,
    ) -> impl Future<Output = io::Result<Self>> {
        Self::new(handle, path)
    }

    fn file_names(&self) -> &[String] {
        &self.file_names
    }

    fn open_entry_by_index(&self, index: usize) -> impl Future<Output = io::Result<Self::File>> {
        self.open_entry_by_index(index)
    }

    fn file_index(&self) -> &FileIndex {
        &self.file_index
    }
}

enum Buffer {
    Empty,
    Pending(futures::future::BoxFuture<'static, std::io::Result<Bytes>>),
    Ready(Bytes),
}

/// An adaption of [`object_store::buffered::BufReader`] that handles segments of a blob
/// as separate files.
pub struct AsyncArchiveFacetReader {
    store: Arc<dyn ObjectStore>,
    target: ObjectMeta,
    start_offset: u64,
    length: u64,
    at: u64,
    buffer: Buffer,
    capacity: usize,
}

impl AsyncArchiveFacetReader {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        target: ObjectMeta,
        start_offset: u64,
        length: u64,
        at: u64,
    ) -> Self {
        Self {
            store,
            target,
            start_offset,
            length,
            at,
            buffer: Buffer::Empty,
            capacity: 1024 * 1024,
        }
    }

    fn poll_fill_buf_impl(
        &mut self,
        cx: &mut Context<'_>,
        amnt: usize,
    ) -> Poll<std::io::Result<&[u8]>> {
        let buf = &mut self.buffer;
        loop {
            match buf {
                Buffer::Empty => {
                    let store = Arc::clone(&self.store);
                    let path = self.target.location.clone();
                    let offset_from = self.start_offset;
                    let start = self.at.min(self.length) as u64 + offset_from;
                    let end =
                        self.at.saturating_add(amnt as u64).min(self.length) as u64 + offset_from;

                    if start == end {
                        return Poll::Ready(Ok(&[]));
                    }

                    *buf = Buffer::Pending(Box::pin(async move {
                        Ok(store.get_range(&path, start..end).await?)
                    }))
                }
                Buffer::Pending(fut) => match ready!(fut.poll_unpin(cx)) {
                    Ok(b) => *buf = Buffer::Ready(b),
                    Err(e) => return Poll::Ready(Err(e)),
                },
                Buffer::Ready(r) => return Poll::Ready(Ok(r)),
            }
        }
    }
}

impl AsyncSeek for AsyncArchiveFacetReader {
    fn start_seek(mut self: Pin<&mut Self>, position: SeekFrom) -> std::io::Result<()> {
        self.at = match position {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => self.length.checked_add_signed(offset).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Seeking {offset} from end of {} byte file would result in overflow",
                        self.length
                    ),
                )
            })?,
            SeekFrom::Current(offset) => self.at.checked_add_signed(offset).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Seeking {offset} from current offset of {} would result in overflow",
                        self.at
                    ),
                )
            })?,
        };
        self.buffer = Buffer::Empty;
        Ok(())
    }

    fn poll_complete(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<u64>> {
        Poll::Ready(Ok(self.at))
    }
}

impl AsyncRead for AsyncArchiveFacetReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // Read the maximum of the internal buffer and `out`
        let to_read = out.remaining().max(self.capacity);
        let r = match ready!(self.poll_fill_buf_impl(cx, to_read)) {
            Ok(buf) => {
                let to_consume = out.remaining().min(buf.len());
                out.put_slice(&buf[..to_consume]);
                self.consume(to_consume);
                Ok(())
            }
            Err(e) => Err(e),
        };
        Poll::Ready(r)
    }
}

impl AsyncBufRead for AsyncArchiveFacetReader {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<&[u8]>> {
        let capacity = self.capacity;
        self.get_mut().poll_fill_buf_impl(cx, capacity)
    }

    fn consume(mut self: Pin<&mut Self>, amt: usize) {
        match &mut self.buffer {
            Buffer::Empty => assert_eq!(amt, 0, "cannot consume from empty buffer"),
            Buffer::Ready(b) => match b.len().cmp(&amt) {
                std::cmp::Ordering::Less => panic!("{amt} exceeds buffer sized of {}", b.len()),
                std::cmp::Ordering::Greater => *b = b.slice(amt..),
                std::cmp::Ordering::Equal => self.buffer = Buffer::Empty,
            },
            Buffer::Pending(_) => panic!("cannot consume from pending buffer"),
        }
        self.at += amt as u64;
    }
}

#[derive(Clone)]
pub struct AsyncArchiveReader<T: AsyncArchiveSource + 'static> {
    archive: T,
    members: Arc<SchemaMetadataManager>,
}

impl<T: AsyncArchiveSource + 'static> AsyncArchiveReader<T> {
    pub fn file_index(&self) -> &FileIndex {
        self.archive.file_index()
    }

    async fn init_from_archive(archive: T) -> io::Result<Self> {
        let mut members = SchemaMetadataManager::default();
        for (i, name) in archive.file_names().iter().enumerate() {
            let tp = archive
                .file_index()
                .iter()
                .find(|s| s.name == *name)
                .map(|s| s.archive_type());
            let metadata = archive.metadata_for_index(i).await.ok();
            let tp = tp.unwrap_or_else(|| if name.ends_with(MzPeakArchiveType::SpectrumDataArrays.tag_file_suffix()) {
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
            });

            if !matches!(tp, MzPeakArchiveType::Other | MzPeakArchiveType::Proprietary) && metadata.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{name} classified as {tp:?} was expected to be a Parquet file, but was not"
                    ),
                ));
            }

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
                MzPeakArchiveType::Other | MzPeakArchiveType::Proprietary => {}
            }
        }
        Ok(Self {
            archive,
            members: Arc::new(members),
        })
    }

    pub async fn from_store_path(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
    ) -> io::Result<Self> {
        let archive = T::from_store_path(store, path).await?;
        Self::init_from_archive(archive).await
    }

    pub async fn from_url(url: &Url) -> io::Result<Self> {
        let (store, path) = parse_url(url)?;
        let store = store.into();
        Self::from_store_path(store, path).await
    }

    pub async fn chromatograms_metadata(
        &self,
    ) -> io::Result<ParquetRecordBatchStreamBuilder<T::File>> {
        if let Some(meta) = self.members.chromatogram_metadata.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone().unwrap()))
                .await
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Chromatogram metadata entry not found",
            ))
        }
    }

    pub async fn chromatograms_data(&self) -> io::Result<ParquetRecordBatchStreamBuilder<T::File>> {
        if let Some(meta) = self.members.chromatogram_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone().unwrap()))
                .await
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Chromatogram data entry not found",
            ))
        }
    }

    pub async fn spectra_data(&self) -> io::Result<ParquetRecordBatchStreamBuilder<T::File>> {
        if let Some(meta) = self.members.spectrum_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone().unwrap()))
                .await
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum data entry not found",
            ))
        }
    }

    pub async fn spectrum_peaks(&self) -> io::Result<ParquetRecordBatchStreamBuilder<T::File>> {
        if let Some(meta) = self.members.peaks_data_arrays.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone().unwrap()))
                .await
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum peak data entry not found",
            ))
        }
    }

    pub async fn spectrum_metadata(&self) -> io::Result<ParquetRecordBatchStreamBuilder<T::File>> {
        if let Some(meta) = self.members.spectrum_metadata.as_ref() {
            self.archive
                .read_index(meta.entry_index, Some(meta.metadata.clone().unwrap()))
                .await
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Spectrum metadata entry not found",
            ))
        }
    }

    pub fn list_files(&self) -> &[String] {
        self.archive.file_names()
    }

    pub fn open_stream(&self, name: &str) -> impl Future<Output = Result<<T as AsyncArchiveSource>::File, io::Error>> {
        self.archive.open_stream(name)
    }
}

#[cfg(test)]
mod test {
    use tokio::io::AsyncReadExt;

    use crate::archive::MzPeakArchiveType;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_local() -> io::Result<()> {
        let store = object_store::local::LocalFileSystem::new_with_prefix(".")?;
        let v = store.path_to_filesystem(&ObjectPath::from("small.mzpeak"))?;
        eprintln!("{}", v.display());

        let handle = AsyncZipArchiveSource::new(Arc::new(store), "small.mzpeak".into()).await?;

        for (i, f) in handle.file_names().iter().enumerate() {
            if f.ends_with(MzPeakArchiveType::SpectrumMetadata.tag_file_suffix()) {
                let dataset = handle.read_index(i, None).await?;
                let meta = dataset.metadata();
                let kv_data = meta.file_metadata().key_value_metadata().unwrap();
                assert!(!kv_data.is_empty())
            }
        }

        assert_eq!(handle.file_index().len(), 4);

        let mut buf = String::new();
        let mut reader = handle.open_stream("mzpeak_index.json").await?;
        reader.read_to_string(&mut buf).await?;
        assert!(buf.len() > 0);

        Ok(())
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    // async fn test_http() -> io::Result<()> {
    //     let store = object_store::http::HttpBuilder::new()
    //         .with_url("http://127.0.0.1:8000/")
    //         .with_client_options(
    //             ClientOptions::new()
    //                 .with_allow_http(true)
    //                 .with_timeout(std::time::Duration::new(10, 0))
    //         )
    //         .build()?;

    //     let path = ObjectPath::from("small.mzpeak");
    //     let handle = AsyncZipArchiveSource::new(Arc::new(store), path).await?;

    //     eprintln!("{:?}",handle.file_names());
    //     Ok(())
    // }
}
