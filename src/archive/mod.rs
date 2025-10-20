mod sync;

mod file_index;

pub use file_index::{DataKind, FileEntry, EntityType, FileIndex};

#[cfg(feature = "async")]
mod object_store_async;

pub use sync::*;

#[cfg(feature = "async")]
pub use object_store_async::{
    AsyncArchiveFacetReader, AsyncArchiveReader, AsyncArchiveSource, AsyncZipArchiveSource,
};
