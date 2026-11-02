# Encoding of explanatory variables ---------------------------------------
# Numeric / logical columns are used as is. Factor / character columns are
# expanded into k - 1 dummy (0/1) columns, the first level being the
# reference. The "specification" built at fit time is stored in the model so
# that new data are encoded in exactly the same way at predict time
# (same columns, same order, same levels).

#' Build the encoding specification from a training data frame
#' @param X data.frame (complete cases).
#' @return list with one element per column: name, type, levels.
#' @noRd
.build_spec <- function(X) {
  lapply(names(X), function(nm) {
    x <- X[[nm]]
    if (is.numeric(x) || is.logical(x)) {
      list(name = nm, type = "numeric")
    } else if (is.factor(x) || is.character(x)) {
      list(name = nm, type = "factor", levels = levels(factor(x)))
    } else {
      stop("Unsupported type for column '", nm, "' (",
           paste(class(x), collapse = "/"),
           "): use numeric, logical, factor or character.", call. = FALSE)
    }
  })
}

#' Check that all columns required by the specification are present
#' @noRd
.check_columns <- function(X, spec) {
  need <- vapply(spec, function(s) s$name, character(1L))
  miss <- setdiff(need, names(X))
  if (length(miss) > 0L) {
    stop("Missing column(s) in 'X': ", paste(miss, collapse = ", "), call. = FALSE)
  }
  invisible(TRUE)
}

#' Encode a data frame into a numeric design matrix
#'
#' @param X data.frame.
#' @param spec specification from `.build_spec()`.
#' @return list(Xm = numeric matrix, assign = original variable name of each
#'   column, bad_rows = logical, TRUE when a factor level was never seen).
#' @noRd
.encode_matrix <- function(X, spec) {
  n <- nrow(X)
  blocks <- vector("list", length(spec))
  assign <- vector("list", length(spec))
  bad_rows <- rep(FALSE, n)
  for (j in seq_along(spec)) {
    s <- spec[[j]]
    x <- X[[s$name]]
    if (s$type == "numeric") {
      v <- if (is.numeric(x) || is.logical(x)) {
        as.numeric(x)
      } else {
        suppressWarnings(as.numeric(as.character(x)))
      }
      m <- matrix(v, nrow = n, ncol = 1L, dimnames = list(NULL, s$name))
      a <- s$name
    } else {
      lv <- s$levels
      k <- length(lv)
      idx <- match(as.character(x), lv)
      bad_rows <- bad_rows | (!is.na(x) & is.na(idx))
      if (k >= 2L) {
        # column l-1 is the indicator of level l (l = 2..k); level 1 = reference
        m <- matrix(as.numeric(idx == rep(seq.int(2L, k), each = n)),
                    nrow = n, ncol = k - 1L)
        colnames(m) <- paste0(s$name, "=", lv[-1L])
        a <- rep(s$name, k - 1L)
      } else {
        m <- matrix(0, nrow = n, ncol = 0L)
        a <- character(0)
      }
    }
    blocks[[j]] <- m
    assign[[j]] <- a
  }
  list(Xm = do.call(cbind, blocks), assign = unlist(assign), bad_rows = bad_rows)
}

#' Detect constant columns (cheap, exact)
#' @noRd
.constant_cols <- function(Xm) {
  vapply(seq_len(ncol(Xm)), function(j) {
    r <- range(Xm[, j])
    r[1L] == r[2L]
  }, logical(1L))
}
