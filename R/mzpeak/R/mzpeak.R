MZPeakSpectrumMetadataFile <- R6::R6Class(
  "MZPeakSpectrumMetadataFile",
  public = list(
    path = NULL,
    meta = NULL,
    file_description = NULL,
    instrument_configuration_list = NULL,
    data_processing_method_list = NULL,
    software_list = NULL,
    sample_list = NULL,
    spectra = NULL,
    scans = NULL,
    precursors = NULL,
    selected_ions = NULL,

    initialize = function(path) {
      logger::log_debug("Loading metadata")
      # Read the entire table into RAM, we will decompose it next
      raw_df <- arrow::read_parquet(path, as_data_frame = FALSE)

      # Store the original connection or location, we may later
      # want it for something else.
      self$path <- path

      # Cache the entire file metadata blob, we will decompose
      # and parse parts of it next
      self$meta <- raw_df$metadata

      # Extract the JSON structures describing the MS run itself
      self$file_description <- jsonlite::fromJSON(self$meta$file_description)
      self$instrument_configuration_list <- jsonlite::fromJSON(self$meta$instrument_configuration_list)
      self$data_processing_method_list <- jsonlite::fromJSON(self$meta$data_processing_method_list)
      self$software_list <- jsonlite::fromJSON(self$meta$software_list)
      self$sample_list <- jsonlite::fromJSON(self$meta$sample_list)

      # Extract the spectrum metadata partition.
      # Drop all rows in this table where the index
      # column is NULL.
      spectra <- raw_df$GetColumnByName("spectrum")
      if (!is.null(spectra)) {
        self$spectra <- chunks_to_table(spectra)
      }

      # Extract the scan partition.
      # Drop all rows in this table where the parent index
      # column is NULL.
      scans <- raw_df$GetColumnByName("scan")
      if (!is.null(scans)) {
        self$scans <- chunks_to_table(scans)
      }

      # Extract the precursor partition.
      # Drop all rows in this table where the parent index
      # column is NULL.
      precursors <- raw_df$GetColumnByName("precursor")
      if (!is.null(precursors)) {
        self$precursors <- chunks_to_table(precursors)
      }

      # Extract the selected ion partition.
      # Drop all rows in this table where the parent index
      # column is NULL.
      selected_ions <- raw_df$GetColumnByName("selected_ion")
      if (!is.null(selected_ions)) {
        self$selected_ions <- chunks_to_table(selected_ions)
      }
    }
  ),
  private = list()
)


length.MZPeakSpectrumMetadataFile <- function(self) {
  length(self$spectra)
}


MZPeakSpectrumDataFile <- R6::R6Class(
  "MZPeakSpectrumDataFile",
  public = list(
    handle = NULL,
    meta = NULL,
    indices = NULL,
    array_metadata = NULL,
    index_bins = NULL,
    mz_delta_models = NULL,
    initialize = function(path) {
      self$handle <- arrow::ParquetFileReader$create(path)
      # Read the file-level metadata off the first row group.
      # The 0th column should be fast to read as it is the
      # spectrum index
      self$meta <- self$handle$ReadRowGroup(0, c(0))$metadata
      # Parse the array metadata from the JSON blob. This will
      # help us understand what kind of data is in each array
      # and un-mangle names, if needed.
      self$array_metadata <- jsonlite::fromJSON(self$meta$spectrum_array_index)
      # Read the spectrum index column from each row group, building
      # up a min-max index for each row group to help reduce work to
      # to read data later.
      self$index_bins <- build_index_bins_direct(self$handle, 0)
      # If present, this will eventually be populated with the parameters
      # fill in NULL-marked positions.
      self$mz_delta_models <- NULL
    },
    # A helper method to determine which row group(s) to search
    row_groups_for_index = function(index) {
      row_groups = self$index_bins[(self$index_bins$min_value <= index) &&
                                   (self$index_bins$max_value >= index), ]$row_group
      row_groups
    },
    # Read the actual signal data associated with a spectrum
    read_spectrum = function(index) {
      if (length(index) > 1) {
        values <- lapply(index, self$read_spectrum)
        names(values) <- index
        return(values)
      }
      index = index - 1
      row_groups = self$row_groups_for_index(index)

      if (length(row_groups) == 1) {
        rg <- self$handle$ReadRowGroup(row_groups[1])
        points <- rg$GetColumnByName("point")
        points <- dplyr::bind_rows(lapply(points$chunks, function(chunk) {
          chunk$Filter(chunk$field(0) == index)$as_vector()
        }))
        k <- dim(points)[2]
        points <- points[, 2:k]
        if (any(is.na(points[, 1]))) {
          model = self$mz_delta_models[[index + 2]]
          points[[1]] <- mzpeak:::.fill_nulls_with_model_or_local(points[[1]], model)
          na_mask <- is.na(points[[2]])
          points[na_mask, 2] = 0
        }
        return(points)
      } else {
        stop("error: not implemented")
      }
    },
    set_mz_delta_models = function(models) {
      self$mz_delta_models = models
    }
  )
)


`[.MZPeakFile` <- function(self, index) {
  self$read_spectrum(index + 1)
}


`[.MZPeakSpectrumDataFile` <- function(self, index) {
  self$read_spectrum(index + 1)
}


derive_id_min_max <- function(array) {
  min_max <- arrow::call_function("min_max", array$View(arrow::int64()))$as_vector()
  return(min_max)
}


build_index_bins_direct <- function(pq_reader, column_index) {
  n <- pq_reader$num_row_groups
  logger::log_debug("Loading bounds for", n, "row groups")
  bounds = lapply(seq(n), function(i) {
    derive_id_min_max(pq_reader$ReadRowGroup(i - 1, 0)$column(0))
  })

  bounds = dplyr::bind_rows(bounds)
  bounds$row_group <- seq(n) - 1
  names(bounds) <- c("min_value", "max_value", "row_group")
  bounds
}


MZPeakFile <- R6::R6Class(
  "MZPeakFile",
  public = list(
    handle = NULL,
    spectrum_metadata = NULL,
    spectrum_data = NULL,
    initialize = function(path) {
      self$handle <- ArchiveHandle$new(path)
      self$spectrum_data = MZPeakSpectrumDataFile$new(self$handle$connect_file("spectra_data.mzpeak"))
      self$spectrum_metadata = MZPeakSpectrumMetadataFile$new(self$handle$connect_file("spectra_metadata.mzpeak"))
      if (any(names(self$spectrum_metadata$spectra) == "mz_delta_model")) {
        self$spectrum_data$set_mz_delta_models(self$spectrum_metadata$spectra$mz_delta_model)
      }
    },
    read_spectrum = function(index) {
      self$spectrum_data$read_spectrum(index)
    },
    query = function() {
      MZPeakQuery$new(self)
    }
  ),
  active = list(
    spectra = function() {
      self$spectrum_metadata$spectra
    },
    scans = function() {
      self$spectrum_metadata$scans
    },
    precursors = function() {
      self$spectrum_metadata$precursors
    },
    selected_ions = function() {
      self$spectrum_metadata$selected_ions
    }
  )
)

length.MZPeakSpectrumDataFile <- function(self) {
  as.numeric(self$meta$spectrum_count)
}

length.MZPeakFile <- function(self) {
  length(self$spectrum_metadata$spectra)
}

dim.MZPeakFile <- function(self) {
  dim(self$spectra)
}


chunks_to_table <- function(chunked_array, mask_column=0) {
  columns_of <- names(chunked_array$type)
  chunks <- lapply(chunked_array$chunks, function(chunk) {
    chunk <- if(chunk$field(mask_column)$null_count > 0) {
      chunk$Filter(!is.na(chunk$field(mask_column)))
    } else {
      chunk
    }
    parts <- chunk$Flatten()
    names(parts) <- columns_of
    do.call(arrow::arrow_table, parts)
  })
  do.call(rbind, chunks)
}
