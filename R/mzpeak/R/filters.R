.find_nulls <- function(dat) {
  indices = which(is.na(dat))
  if (indices[1] != 0) {
    indices <- c(0, indices)
  }
  n = length(dat)
  if (indices[length(indices)] != n) {
    indices <- c(indices, n)
  }

  matrix(indices, ncol=2, byrow = TRUE)
}


estimate_local_median <- function(x) {
  dx  = diff(x[!is.na(x)])
  median_dx = median(dx)
  median(dx[dx <= median_dx])
}


.fill_nulls_with_model_or_local <- function(dat, model) {
  inds <- .find_nulls(dat)
  unlist(apply(inds, 1, function(ii) {
    chunk = dat[ii[1]:ii[2]]
    has_real = chunk[!is.na(chunk)]
    if (length(has_real) == 1) {
      dx = chunk[2] ^ seq(length(model))
      chunk[1] = chunk[2] - dx
      chunk[3] = chunk[2] + dx
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
}
