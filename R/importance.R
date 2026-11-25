# Relevance of the explanatory variables -----------------------------------
# The unit of analysis is the ORIGINAL variable: a categorical predictor with
# k levels is a block of m = k - 1 dummy columns, tested as a whole.
#
# Partial Wilks' lambda of variable v (block of m columns) given the p - m
# other columns of the model:
#
#   Lambda(v | others) = Lambda(full) / Lambda(others)
#                      = det((T^-1)_vv) / det((E^-1)_vv)
#
# The second form only needs the inverses of E and T that the solver already
# computed (block-inverse identity det(A^-1)_vv = det(A_others) / det(A)), so
# the whole table costs O(p^2) per variable and never re-fits a model.
#
# Lambda(v | others) ~ U(m, K - 1, n - K - (p - m)), tested with Rao's F
# ("F to remove"). For m = 1 the F is exact, with (K - 1, n - K - p + 1) df.
#
# Tolerance = share of the within-class variability of a variable that is NOT
# explained by the other variables (values close to 0 flag redundancy).
#   numeric variable (m = 1): 1 - R^2 of the column regressed on the others,
#                             within classes = 1 / (E_jj (E^-1)_jj);
#   block (m > 1)           : generalised tolerance
#                             det(E_vv.others) / det(E_vv) = 1 / (det(E_vv) det((E^-1)_vv))
#                             = prod_i (1 - rho_i^2), rho_i the canonical
#                             correlations between the block and the others.
# Unlike the smallest tolerance of the dummy columns, the generalised value
# does not depend on the reference level chosen for the dummy coding, and it
# reduces to the usual tolerance when m = 1.

#' Per-variable partial statistics
#'
#' @param Einv,Tinv inverses of the within-class and total SSCP matrices
#'   (p x p, standardised scale).
#' @param e_diag diagonal of E on the same scale (all ones when standardised);
#'   only used for single columns, blocks use the full E recovered from Einv.
#' @param assign original variable of each column (length p).
#' @param n,K number of observations and of classes.
#' @return data.frame(variable, n_columns, partial_wilks, F, df1, df2,
#'   p_value, tolerance), one row per original variable, in model order.
#' @noRd
.partial_table <- function(Einv, Tinv, e_diag, assign, n, K) {
  p <- length(assign)
  vars <- unique(assign)
  tol <- 1 / (e_diag * diag(Einv))         # 1 - R^2 of the column on the others
  has_block <- any(duplicated(assign))
  E <- if (has_block) chol2inv(chol(Einv)) else NULL    # back to E (p x p, cheap)
  res <- matrix(NA_real_, length(vars), 7L)
  for (i in seq_along(vars)) {
    idx <- which(assign == vars[i])
    m <- length(idx)
    lam <- if (m == 1L) {
      Tinv[idx, idx] / Einv[idx, idx]
    } else {
      exp(.logdet(Tinv[idx, idx, drop = FALSE]) - .logdet(Einv[idx, idx, drop = FALSE]))
    }
    lam <- min(max(lam, 0), 1)             # guard against rounding (lambda in [0, 1])
    rf <- .rao_F(lam, m, K - 1L, n - K - (p - m))
    tl <- if (m == 1L) {
      tol[idx]
    } else {
      exp(-.logdet(E[idx, idx, drop = FALSE]) - .logdet(Einv[idx, idx, drop = FALSE]))
    }
    res[i, ] <- c(m, lam, rf[["F"]], rf[["df1"]], rf[["df2"]], rf[["p_value"]],
                  min(max(tl, 0), 1))
  }
  out <- data.frame(variable = vars, res, stringsAsFactors = FALSE)
  names(out) <- c("variable", "n_columns", "partial_wilks", "F", "df1", "df2",
                  "p_value", "tolerance")
  out
}
