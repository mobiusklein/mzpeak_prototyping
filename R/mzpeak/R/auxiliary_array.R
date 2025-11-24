ZLIB_COMPRESSION_CURIE = "MS:1000574"
NO_COMPRESSION_CURIE = "MS:1000576"

INT32_TYPE = "MS:1000519"
FLOAT32_TYPE = "MS:1000521"
INT64_TYPE = "MS:1000522"
FLOAT64_TYPE = "MS:1000523"
ASCII_TYPE = "MS:1001479"

.decode_int <- function(data, width) {
  readBin(data, what = "integer", size = width, n = length(data) / width)
}

.decode_float <- function(data, width) {
  readBin(data, what = "double", size = width, n = length(data) / width)
}

.decode_ascii <- function(data) {
  ii <- c(0, which(data == 0))
  unlist(purrr::map2(ii[1:length(ii) - 1], ii[2:length(ii)], function(i, j) {
    rawToChar(data[seq(i : j) - 1])
  }))
}

decode_auxiliary_array <- function(arr) {
  data <- as.raw(arr$data[[1]])
  compression <- arr$compression
  dtype <- arr$data_type
  name_param <- arr$name
  if (name_param$name == "non-standard data array") {
    name <- name_param$value$string
  } else {
    name <- name_param$name
  }
  if (compression == NO_COMPRESSION_CURIE) {
    data <- data
  } else if (compression == ZLIB_COMPRESSION_CURIE) {
    stop("zlib compression not yet supported")
  } else {
    stop(paste("Compression method", compression, "not supported or recognized"))
  }

  if (dtype == INT32_TYPE) {
    data <- .decode_int(data, 4)
  } else if (dtype == INT64_TYPE) {
    data <- .decode_int(data, 8)
  } else if (dtype == FLOAT32_TYPE) {
    data <- .decode_float(data, 4)
  } else if (dtype == FLOAT64_TYPE) {
    data <- .decode_float(data, 8)
  } else if (dtype == ASCII_TYPE) {
    data <- .decode_ascii(data)
  } else {
    stop(paste("Data type", dtype, "not supported or recognized"))
  }

  list(name=name, data=data)
}
