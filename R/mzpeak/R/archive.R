
connection_segment <- function(path, name) {
  blocks <- zip::zip_list(path)
  mask <- blocks$filename == name
  if(!any(mask)) {
    error(
      paste("Failed to locate an archive member with the name", name, "in archive", path)
    )
  }
  block <- blocks[mask, ]
  offset <- block$offset
  size <- block$uncompressed_size
  handle = .Call("_make_connection_segment",
                 path,
                 name,
                 offset,
                 size)
  return(handle);
}

ARCHIVE_TYPES = factor(c("ZIP", "UNPACKED", "ARROWFS"))
ZIP_ARCHIVE = ARCHIVE_TYPES[1]
UNPACKED_ARCHIVE = ARCHIVE_TYPES[2]
ARROW_FS = ARCHIVE_TYPES[3]

ArchiveHandle <- R6::R6Class(
  "ArchiveHandle",
  public=list(
    root=NULL,
    archive_type=NULL,
    initialize = function(path) {
      self$root <- path
      if (!file.exists(path)) {
        stop(paste(path, "does not exist, cannot open"))
      }
      if (dir.exists(path)) {
        self$archive_type <- UNPACKED_ARCHIVE
      } else {
        self$archive_type <- ZIP_ARCHIVE
      }
    },
    has_file = function(name) {
      if (self$archive_type == UNPACKED_ARCHIVE) {
        return(file.exists(file.path(self$root, name)))
      } else {
        members <- zip::zip_list(self$root)
        return(any(members$filename == name))
      }
    },
    connect_file = function(name) {
      if (self$archive_type == UNPACKED_ARCHIVE) {
        return(file.path(self$root, name))
      } else {
        x <- connection_segment(self$root, name)
        open(x)
        return(x)
      }
    }
  )
)

