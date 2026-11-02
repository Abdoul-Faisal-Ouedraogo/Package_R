# Computational core --------------------------------------------------------
# Everything is derived from a handful of sufficient statistics computed in a
# single pass over the data (rowsum + one crossprod), then handled with
# Cholesky factorisations. No loop over observations, no explicit inversion
# of a general matrix.
#
# Notation: n observations, K classes, p encoded columns.
#   means : K x p class means          grand : overall mean (length p)
#   E     : p x p within-class SSCP    B     : p x p between-class SSCP
#   T     : E + B (total SSCP)         W     : E / (n - K) pooled covariance

#' Sufficient statistics of a (design matrix, class) pair
#' @param Xm numeric matrix n x p (no NA).
#' @param y factor with no empty level.
#' @return list(n, K, cnt, means, grand, E, B, T).
#' @noRd
.suff_stats <- function(Xm, y) {
  n <- nrow(Xm)
  K <- nlevels(y)
  g <- as.integer(y)
  cnt <- tabulate(g, nbins = K)
  sums <- rowsum(Xm, g, reorder = TRUE)           # K x p, rows ordered by level
  means <- sums / cnt                             # row k divided by cnt[k]
  dimnames(means) <- list(levels(y), colnames(Xm))
  grand <- colSums(sums) / n
  Xc <- Xm - means[g, , drop = FALSE]             # centred within class
  E <- crossprod(Xc)
  D <- sweep(means, 2L, grand, "-") * sqrt(cnt)   # sqrt(n_k) (m_k - m)
  B <- crossprod(D)
  list(n = n, K = K, cnt = cnt, means = means, grand = grand,
       E = E, B = B, T = E + B)
}

#' Restrict sufficient statistics to a subset of columns
#' @noRd
.subset_stats <- function(st, idx) {
  st$means <- st$means[, idx, drop = FALSE]
  st$grand <- st$grand[idx]
  st$E <- st$E[idx, idx, drop = FALSE]
  st$B <- st$B[idx, idx, drop = FALSE]
  st$T <- st$T[idx, idx, drop = FALSE]
  if (!is.null(st$scale)) st$scale <- st$scale[idx]
  st
}

#' Express the sufficient statistics on a standardised scale
#'
#' LDA is invariant to a rescaling of the columns. Working with E / (s s')
#' (unit diagonal: the within-class correlation matrix) keeps the Cholesky
#' factorisations and the stepwise computations well conditioned even when
#' variables have very different units (e.g. 1e3 versus 1e-2). The
#' coefficients are transformed back to the original units in `.lda_solve()`.
#' @param s positive scale of each column (square root of diag(E)).
#' @noRd
.scale_stats <- function(st, s) {
  ss <- tcrossprod(s)
  st$means <- sweep(st$means, 2L, s, "/")
  st$grand <- st$grand / s
  st$E <- st$E / ss
  st$B <- st$B / ss
  st$T <- st$T / ss
  st$scale <- s
  st
}

#' Columns to keep so that the within-class matrix is non singular
#'
#' Drops columns that are (numerically) constant within classes and columns
#' that are linear combinations of previous ones. Works on the p x p
#' correlation matrix of E (cheap) with the same limited-pivoting QR as lm().
#' @return integer indices of the retained columns (increasing order).
#' @noRd
.independent_columns <- function(E, Tm, tol = 1e-7) {
  d <- diag(E)
  td <- diag(Tm)
  ok <- which(is.finite(d) & td > 0 & d > 1e-12 * td)
  if (length(ok) == 0L) return(integer(0L))
  R <- stats::cov2cor(E[ok, ok, drop = FALSE])
  q <- qr(R, tol = tol)
  ok[sort(q$pivot[seq_len(q$rank)])]
}

#' log-determinant of a symmetric positive definite matrix
#' @noRd
.logdet <- function(A) as.numeric(determinant(A, logarithm = TRUE)$modulus)

#' Rao's F approximation of Wilks' lambda
#'
#' Lambda ~ U(p, q, N): p variables, q hypothesis df (= K - 1), N error df.
#' Same formula as summary.manova(test = "Wilks"); for p = 1 it is the
#' ANOVA F, for q = 1 it is Hotelling's T2 test.
#' @return named numeric: F, df1, df2, p_value.
#' @noRd
.rao_F <- function(lambda, p, q, N) {
  lambda <- min(max(lambda, 0), 1)
  d <- p^2 + q^2 - 5
  s <- if (d > 0) sqrt((p^2 * q^2 - 4) / d) else 1
  df1 <- p * q
  df2 <- s * (N - (p - q + 1) / 2) - (p * q - 2) / 2
  if (!is.finite(df2) || df2 <= 0) {
    return(c(F = NA_real_, df1 = df1, df2 = df2, p_value = NA_real_))
  }
  Fval <- (lambda^(-1 / s) - 1) * df2 / df1
  c(F = Fval, df1 = df1, df2 = df2,
    p_value = stats::pf(Fval, df1, df2, lower.tail = FALSE))
}

#' Solve the LDA problem from sufficient statistics
#'
#' Classification functions: delta_k(x) = x' a_k + b_k with
#' a_k = W^-1 m_k and b_k = log(pi_k) - 0.5 m_k' W^-1 m_k.
#' Canonical axes: generalised eigenproblem B a = lambda E a, obtained from a
#' K x p SVD (rank <= K - 1), scaled so that a' W a = 1 (as MASS::lda).
#' `st` is expected on the standardised scale (see `.scale_stats()`); `Einv`
#' and `Tinv` are returned on that scale.
#' @noRd
.lda_solve <- function(st, priors) {
  n <- st$n; K <- st$K; p <- ncol(st$E)
  Re <- tryCatch(chol(st$E), error = function(e) {
    stop("The within-class covariance matrix is not positive definite ",
         "(near-collinear predictors). Remove redundant variables or use ",
         "variable selection.", call. = FALSE)
  })
  Rt <- chol(st$T)
  Einv <- chol2inv(Re)
  Tinv <- chol2inv(Rt)

  # classification functions (computed on the standardised scale, then
  # expressed in the original units: x' a_z / s)
  s <- if (is.null(st$scale)) rep(1, p) else st$scale
  coef_z <- (n - K) * (Einv %*% t(st$means))               # p x K  (W^-1 m_k)
  intercept <- log(priors) - 0.5 * rowSums(st$means * t(coef_z))
  coef <- coef_z / s
  dimnames(coef) <- list(colnames(st$means), rownames(st$means))
  names(intercept) <- rownames(st$means)

  # Wilks' lambda = det(E) / det(T), through the Cholesky factors
  wilks <- exp(2 * sum(log(diag(Re))) - 2 * sum(log(diag(Rt))))
  test <- .rao_F(wilks, p, K - 1L, n - K)

  # canonical discriminant axes
  r <- min(p, K - 1L)
  Dw <- sweep(st$means, 2L, st$grand, "-") * sqrt(st$cnt)  # K x p
  Gt <- backsolve(Re, t(Dw), transpose = TRUE)             # p x K  (R^-T Dw')
  sv <- svd(Gt, nu = r, nv = 0L)
  eig <- sv$d[seq_len(r)]^2                                # eigenvalues of E^-1 B
  axes <- backsolve(Re, sv$u[, seq_len(r), drop = FALSE]) * sqrt(n - K) / s
  for (j in seq_len(r)) {                                  # deterministic sign
    if (axes[which.max(abs(axes[, j])), j] < 0) axes[, j] <- -axes[, j]
  }
  dimnames(axes) <- list(colnames(st$means), paste0("LD", seq_len(r)))

  list(coef = coef, intercept = intercept, wilks = wilks, test = test,
       axes = axes, eigenvalues = eig,
       canonical_cor = sqrt(eig / (1 + eig)),
       Einv = Einv, Tinv = Tinv)
}

#' Per-variable partial statistics
#'
#' Partial Wilks' lambda of variable v (a block of m encoded columns) given
#' all the others:  det((T^-1)_vv) / det((E^-1)_vv)  (a scalar ratio when m = 1).
#' Rao's F then tests its contribution ("F to remove").
#' @noRd
.partial_table <- function(Einv, Tinv, e_diag, assign, n, K) {
  p <- length(assign)
  vars <- unique(assign)
  tol <- 1 / (e_diag * diag(Einv))         # 1 - R^2 of the column on the others
  res <- matrix(NA_real_, length(vars), 7L)
  for (i in seq_along(vars)) {
    idx <- which(assign == vars[i])
    m <- length(idx)
    lam <- if (m == 1L) {
      Tinv[idx, idx] / Einv[idx, idx]
    } else {
      exp(.logdet(Tinv[idx, idx, drop = FALSE]) - .logdet(Einv[idx, idx, drop = FALSE]))
    }
    rf <- .rao_F(lam, m, K - 1L, n - K - (p - m))
    res[i, ] <- c(m, lam, rf[["F"]], rf[["df1"]], rf[["df2"]], rf[["p_value"]],
                  min(tol[idx]))
  }
  out <- data.frame(variable = vars, res, stringsAsFactors = FALSE)
  names(out) <- c("variable", "n_columns", "partial_wilks", "F", "df1", "df2",
                  "p_value", "tolerance")
  out
}

#' Per-class precision / recall / F1 from a confusion matrix
#' @noRd
.class_metrics <- function(cm) {
  cm <- unclass(cm)
  tp <- diag(cm)
  n_true <- rowSums(cm)
  n_pred <- colSums(cm)
  recall <- ifelse(n_true > 0, tp / n_true, NA_real_)
  precision <- ifelse(n_pred > 0, tp / n_pred, NA_real_)
  f1 <- ifelse(is.na(recall) | is.na(precision) | (recall + precision) == 0,
               NA_real_, 2 * recall * precision / (recall + precision))
  data.frame(class = rownames(cm), n = as.integer(n_true), recall = as.numeric(recall),
             precision = as.numeric(precision), f1 = as.numeric(f1),
             row.names = NULL, stringsAsFactors = FALSE)
}
