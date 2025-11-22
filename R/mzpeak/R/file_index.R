
FileIndex <- R6::R6Class(
  "FileIndexEntry",
  public = list(
    files = NULL,
    metadata = NULL,
    initialize = function(path) {
      data <- jsonlite::fromJSON(path)
      self$files <- data$files
      self$metadata <- data$metadata
    }
  )
)
