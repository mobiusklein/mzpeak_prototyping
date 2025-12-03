.find_nulls <- function(dat) {
  indices = which(is.na(dat))
  if (indices[1] != 1) {
    indices <- c(1, indices)
  }
  n = length(dat)
  if (indices[length(indices)] != n) {
    indices <- c(indices, n)
  }

  matrix(indices, ncol = 2, byrow = TRUE)
}


estimate_local_median <- function(x) {
  dx  = diff(x[!is.na(x)])
  median_dx = median(dx)
  median(dx[dx <= median_dx])
}


.fill_nulls_with_model_or_local <- function(dat, delta_model) {
  inds <- .find_nulls(dat)
  filled <- unlist(apply(inds, 1, function(ii) {
    chunk = dat[ii[1]:ii[2]]
    has_real = chunk[!is.na(chunk)]
    if (length(has_real) == 1) {
      n <- length(chunk)
      if (n == 2) {
        if (is.na(chunk[1])) {
          dx = (chunk[2]^seq(length(delta_model)))
          chunk[1] <- chunk[2] - dx
        } else {
          dx = (chunk[1]^seq(length(delta_model)))
          chunk[2] <- chunk[1] + dx
        }
      } else if (n == 3) {
        dx = (chunk[2]^seq(length(delta_model)))
        chunk[1] = chunk[2] - dx
        chunk[3] = chunk[2] + dx
      } else {
        stop("Failed to fill chunk")
      }
      chunk
    } else {
      n = length(chunk)
      dx = estimate_local_median(has_real)
      if (is.na(chunk[1])) {
        chunk[1] = chunk[2] - dx
      }
      if (is.na(chunk[n])) {
        chunk[n] = chunk[n - 1] + dx
      }
      chunk
    }
  }))
  filled
}


DELTA_ENCODING_CURIE = "MS:1003089"
NO_COMPRESSION_CURIE = "MS:1000576"
NUMPRESS_LINEAR_CURIE = "MS:1002312"

NUMPRESS_SLOF_CURIE = "MS:1002314"
NUMPRESS_PIC_CURIE = "MS:1002313"

#' @description
#' decode deltas in the presence of `NULL` values.
#' @details This method is equivalent to `.delta_decode` save that it
#'  will properly handle NULL values, at the cost of being slower and
#'  requiring interpolation to fill the gaps. See `.delta_decode` for
#'  a description of the parameters.
#' @param start(numeric(1))
#' @param end(numeric(1))
#' @param values(numeric)
#' @param model(numeric)
#' @param has_null(logical) \cr
#'    Which positions in `values` are NULL
.delta_decode_null <- function(start, end, values, model, has_null) {
  origin <- start
  storage <- numeric(1 + length(values))
  i <- 1
  if (has_null[1]) {
    if (has_null[2]) {
      storage[i] <- start
      i <- i + 1
    }
    start <- NA
  } else {
    storage[i] <- start
    i <- i + 1
  }
  last <- start
  for (item in values) {
    if (!is.na(item)) {
      if (!is.na(last)) {
        last <- item + last
        storage[i] <- last
        i <- i + 1
      } else {
        storage[i] <- item
        i <- i + 1
        last <- item
      }
    } else {
      storage[i] <- item
      last <- item
      i <- i + 1
    }
  }
  if (i == length(storage)) {
    storage <- storage[seq_len(i - 1)]
  }
  storage <- .fill_nulls_with_model_or_local(storage, model)
  return(storage)
}

#' @description
#' Decode a delta-encoded chunk. This is a helper method for `.decode_chunk`
#'
#' @param start(numeric(1)) \cr
#'    The starting value for this chunk. All values are relative
#'    to this.
#' @param end(numeric(1)) \cr
#'    The ending value for this chunk. No value will be greater
#'    than this.
#' @param values(numeric) \cr
#'    The deltas of this chunk, w.r.t. `start`
#' @param model(numeric) \cr
#'    The model to use to fill null values if any are present,
#'    which will signal calling `.delta_decode_null` instead.
.delta_decode <- function(start, end, values, model) {
  has_null <- is.na(values)
  if (any(has_null)) {
    return(.delta_decode_null(start, end, values, mode, has_null))
  }
  storage <- numeric(1 + length(values)) + start
  idx <- seq(length(values)) + 1
  if (length(values) > 0) {
    storage[idx] <- storage[idx] + cumsum(values)
  }
  storage
}

#' @description
#' Decode a single row in the chunk layout.
#' @param ... \cr
#'    The columns of the chunk row to be decoded
#' @param delta_model(numeric) \cr
#'    The delta spacing model to use to fill null values
#'    in chunks. This may be NULL itself, which will mean
#'    no null filling.
#' @param array_metadata(data.frame) \cr
#'    A data.frame describing the arrays that may be present
#'    in the chunk. These will be used to decide how to handle
#'    nullity in any array that is not the primary axis, and to
#'    derive any naming changes required.
.decode_chunk <- function(..., delta_model, array_metadata) {
  dat <- list(...)
  encoding <- dat$chunk_encoding
  index <- dat[[1]]
  names_of <- names(dat)

  values_name <- names_of[grepl("_chunk_values", x = names_of, fixed = TRUE)]
  axis_name <- sub(x = values_name,
                   pattern = "_chunk_values",
                   replacement = "")
  start_name <- sub(pattern = "_chunk_values",
                    replacement = "_chunk_start",
                    x = values_name)
  end_name <- sub(pattern = "_chunk_values",
                  replacement = "_chunk_end",
                  x = values_name)

  if (encoding == DELTA_ENCODING_CURIE) {
    values <- .delta_decode(dat[[start_name]], dat[[end_name]], dat[[values_name]], delta_model)
  }
  else if (encoding == NO_COMPRESSION_CURIE) {
    values <- dat[[values_name]]
  }
  else if (encoding == NUMPRESS_LINEAR_CURIE) {
    stop("Numpress support not yet implemented")
  }
  else {
    stop(paste("Don't know how to use "))
  }
  names_to_skip <- c(names_of[1], values_name, start_name, end_name, "chunk_encoding")
  result <- list()
  result[[axis_name]] <- values
  for(name in names_of) {
    if (name %in% names_to_skip || is.na(name)) {
      next
    }
    vals <- dat[[name]]
    array_meta <- array_metadata[array_metadata$path_name == name, ]
    if (array_meta$array_name == "intensity array") {
      vals[is.na(vals)] = 0
    }
    result[[name]] <- vals
  }
  result <- as.data.frame(result)
  result
}


#' @description
#' Decode a batch of chunk layout data.
#' @param dat(data.frame) \cr
#'    The actual data to decode. It should have all the minimal columns
#'    the chunk layout requires.
#' @param delta_model(numeric) \cr
#'    The m/z delta model to use to fill any null values in the data as
#'    it is read from the data file. This may be NULL
#' @param array_metadata(data.frame) \cr
#'    A description of the arrays present in the data file, defining their
#'    names, types, and other useful descriptions
#' @return [data.frame]
decode_chunks_for <- function(dat, delta_model, array_metadata) {
  x <- purrr::pmap_df(dat, function(...) {
    .decode_chunk(..., delta_model=delta_model, array_metadata=array_metadata$entries)
  })
}
