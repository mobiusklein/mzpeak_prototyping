mod sync;

#[cfg(feature = "async")]
mod object_store_async;

pub use sync::*;

#[cfg(feature = "async")]
pub use object_store_async::{
    AsyncArchiveFacetReader,
    AsyncArchiveSource,
    AsyncZipArchiveSource,
    ObjectStoreReader
};