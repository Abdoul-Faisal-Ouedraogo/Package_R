# Evaluation metrics ---------------------------------------------------------
# All metrics are computed from a confusion matrix with the OBSERVED classes in
# rows and the PREDICTED classes in columns (same convention as
# `table(Observed = y, Predicted = pred)`).
#
# Per class k (one-versus-rest):
#   recall_k    = n_kk / n_k.          (sensitivity, row-wise)
#   precision_k = n_kk / n_.k          (positive predictive value, column-wise)
#   F1_k        = 2 recall_k precision_k / (recall_k + precision_k)
# Global:
#   accuracy          = sum_k n_kk / n
#   balanced accuracy = mean_k recall_k          (insensitive to class sizes)
#   macro F1          = mean_k F1_k
#   weighted F1       = sum_k (n_k. / n) F1_k
#   Cohen's kappa     = (p_o - p_e) / (1 - p_e),  p_e = sum_k n_k. n_.k / n^2

#' Validate and align a confusion matrix
#'
#' Accepts a `table` or a numeric matrix with observed classes in rows and
#' predicted classes in columns. When both dimensions are named, the columns
#' are reordered to match the rows; predicted classes that never occur among
#' the observed ones (or the reverse) are added as zero rows / columns so that
#' the result is always square.
#' @return A square numeric matrix with identical row and column names.
#' @noRd
.check_confusion <- function(cm) {
  if (!(is.matrix(cm) || inherits(cm, "table")) || length(dim(cm)) != 2L) {
    stop("'confusion' must be a two-dimensional table or matrix ",
         "(observed classes in rows, predicted classes in columns).", call. = FALSE)
  }
  M <- unclass(cm)
  dn <- dimnames(M)
  storage.mode(M) <- "double"
  if (any(!is.finite(M)) || any(M < 0)) {
    stop("'confusion' must contain non-negative finite counts.", call. = FALSE)
  }
  rn <- dn[[1L]]
  cn <- if (length(dn) >= 2L) dn[[2L]] else NULL
  if (is.null(rn) || is.null(cn)) {
    if (nrow(M) != ncol(M)) {
      stop("An unnamed 'confusion' matrix must be square.", call. = FALSE)
    }
    if (is.null(rn)) rn <- if (!is.null(cn)) cn else paste0("class", seq_len(nrow(M)))
    dimnames(M) <- list(rn, rn)
    return(M)
  }
  labs <- union(rn, cn)
  out <- matrix(0, length(labs), length(labs), dimnames = list(labs, labs))
  out[rn, cn] <- M
  names(dimnames(out)) <- names(dn)
  out
}

#' Per-class recall, precision and F1 from a confusion matrix
#' @param cm confusion matrix (observed in rows).
#' @return data.frame(class, n, recall, precision, f1). `n` is the number of
#'   observations of the class; metrics that are undefined (0 / 0) are `NA`.
#' @noRd
.class_metrics <- function(cm) {
  cm <- .check_confusion(cm)
  tp <- diag(cm)
  n_true <- rowSums(cm)
  n_pred <- colSums(cm)
  recall <- ifelse(n_true > 0, tp / n_true, NA_real_)
  precision <- ifelse(n_pred > 0, tp / n_pred, NA_real_)
  f1 <- ifelse(is.na(recall) | is.na(precision) | (recall + precision) == 0,
               ifelse(!is.na(recall) & !is.na(precision), 0, NA_real_),
               2 * recall * precision / (recall + precision))
  data.frame(class = rownames(cm), n = as.integer(n_true), recall = as.numeric(recall),
             precision = as.numeric(precision), f1 = as.numeric(f1),
             row.names = NULL, stringsAsFactors = FALSE)
}

#' Global metrics from a confusion matrix
#' @param cm confusion matrix (observed in rows).
#' @return Named numeric vector: n, accuracy, error_rate, balanced_accuracy,
#'   macro_precision, macro_recall, macro_f1, weighted_f1, kappa.
#' @noRd
.global_metrics <- function(cm) {
  cm <- .check_confusion(cm)
  n <- sum(cm)
  if (n == 0) {
    stop("'confusion' is empty: no observation was classified.", call. = FALSE)
  }
  pc <- .class_metrics(cm)
  present <- pc$n > 0                         # classes actually observed
  acc <- sum(diag(cm)) / n
  pe <- sum(rowSums(cm) * colSums(cm)) / n^2  # chance agreement
  kappa <- if (pe < 1) (acc - pe) / (1 - pe) else NA_real_
  f1_obs <- pc$f1[present]
  c(n = n,
    accuracy = acc,
    error_rate = 1 - acc,
    balanced_accuracy = mean(pc$recall[present]),
    macro_precision = mean(pc$precision[present], na.rm = TRUE),
    macro_recall = mean(pc$recall[present]),
    macro_f1 = mean(ifelse(is.na(f1_obs), 0, f1_obs)),
    weighted_f1 = sum(pc$n[present] * ifelse(is.na(f1_obs), 0, f1_obs)) / n,
    kappa = kappa)
}
