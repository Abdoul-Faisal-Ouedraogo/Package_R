# Stepwise variable selection by Wilks' lambda -----------------------------
# Works only on the p x p matrices E and T (no pass over the data).
# The unit of selection is the ORIGINAL variable: a categorical predictor is a
# block of m dummy columns that enter / leave together.
#
# Entering variable v given the model S:
#   Lambda(v | S) = det(E_vv|S) / det(T_vv|S)   (Schur complements)
#   ~ U(m, K - 1, n - K - |S|)  -> Rao's F, p-value.
# Leaving variable v given the model S:
#   Lambda(v | S \ v) = det((T_S^-1)_vv) / det((E_S^-1)_vv).

.schur <- function(M, B, S, Minv) {
  A <- M[B, B, drop = FALSE]
  if (length(S) == 0L) return(A)
  C <- M[B, S, drop = FALSE]
  A - C %*% Minv %*% t(C)
}

#' @param E,Tm within-class and total SSCP (p x p).
#' @param assign original variable of each column.
#' @param method "forward", "backward" or "stepwise".
#' @return list(selected = variable names, history = data.frame).
#' @noRd
.select_variables <- function(E, Tm, assign, n, K, method,
                              alpha_enter, alpha_stay) {
  vars <- unique(assign)
  cols_of <- lapply(vars, function(v) which(assign == v))
  names(cols_of) <- vars
  cols <- function(v) as.integer(unlist(cols_of[v], use.names = FALSE))
  q <- K - 1L

  wilks_of <- function(S) {
    if (length(S) == 0L) return(1)
    exp(.logdet(E[S, S, drop = FALSE]) - .logdet(Tm[S, S, drop = FALSE]))
  }

  enter_stats <- function(in_model, cand) {
    S <- cols(in_model)
    Einv <- if (length(S)) solve(E[S, S, drop = FALSE]) else NULL
    Tinv <- if (length(S)) solve(Tm[S, S, drop = FALSE]) else NULL
    out <- matrix(NA_real_, length(cand), 3L,
                  dimnames = list(cand, c("lambda", "F", "p")))
    for (i in seq_along(cand)) {
      B <- cols_of[[cand[i]]]
      ES <- .schur(E, B, S, Einv)
      TS <- .schur(Tm, B, S, Tinv)
      # skip candidates (almost) collinear with the current model
      if (min(diag(ES) / diag(E[B, B, drop = FALSE])) < 1e-8) next
      lam <- exp(.logdet(ES) - .logdet(TS))
      rf <- .rao_F(lam, length(B), q, n - K - length(S))
      out[i, ] <- c(lam, rf[["F"]], rf[["p_value"]])
    }
    out
  }

  leave_stats <- function(in_model) {
    S <- cols(in_model)
    Einv <- solve(E[S, S, drop = FALSE])
    Tinv <- solve(Tm[S, S, drop = FALSE])
    out <- matrix(NA_real_, length(in_model), 3L,
                  dimnames = list(in_model, c("lambda", "F", "p")))
    for (i in seq_along(in_model)) {
      pos <- match(cols_of[[in_model[i]]], S)
      m <- length(pos)
      lam <- if (m == 1L) {
        Tinv[pos, pos] / Einv[pos, pos]
      } else {
        exp(.logdet(Tinv[pos, pos, drop = FALSE]) - .logdet(Einv[pos, pos, drop = FALSE]))
      }
      rf <- .rao_F(lam, m, q, n - K - (length(S) - m))
      out[i, ] <- c(lam, rf[["F"]], rf[["p_value"]])
    }
    out
  }

  in_model <- if (method == "backward") vars else character(0L)
  history <- list()
  add_hist <- function(action, v, F, p) {
    history[[length(history) + 1L]] <<- data.frame(
      step = length(history) + 1L, action = action, variable = v,
      F = F, p_value = p, wilks = wilks_of(cols(in_model)),
      stringsAsFactors = FALSE)
  }
  max_iter <- 2L * length(vars) + 5L
  iter <- 0L

  if (method %in% c("forward", "stepwise")) {
    repeat {
      iter <- iter + 1L
      if (iter > max_iter) break
      cand <- setdiff(vars, in_model)
      if (length(cand) == 0L) break
      st <- enter_stats(in_model, cand)
      best <- order(st[, "p"], st[, "lambda"])[1L]
      if (is.na(st[best, "p"])) break
      passes <- st[best, "p"] < alpha_enter
      if (!passes && length(in_model) > 0L) break
      if (!passes) {
        warning("No variable passed the entry threshold (alpha_enter = ", alpha_enter,
                "); keeping the single best variable.", call. = FALSE)
      }
      v <- cand[best]
      in_model <- c(in_model, v)
      add_hist("enter", v, st[best, "F"], st[best, "p"])
      if (!passes) break
      if (method == "stepwise" && length(in_model) >= 2L) {
        lv <- leave_stats(in_model)
        lv[v, ] <- NA_real_                 # never remove the variable just entered
        worst <- which.max(lv[, "p"])
        if (length(worst) == 1L && lv[worst, "p"] > alpha_stay) {
          vr <- in_model[worst]
          in_model <- setdiff(in_model, vr)
          add_hist("remove", vr, lv[worst, "F"], lv[worst, "p"])
        }
      }
    }
  } else {                                  # backward elimination
    repeat {
      iter <- iter + 1L
      if (iter > max_iter || length(in_model) <= 1L) break
      lv <- leave_stats(in_model)
      worst <- which.max(lv[, "p"])
      if (length(worst) == 0L || lv[worst, "p"] <= alpha_stay) break
      vr <- in_model[worst]
      in_model <- setdiff(in_model, vr)
      add_hist("remove", vr, lv[worst, "F"], lv[worst, "p"])
    }
  }

  hist_df <- if (length(history)) {
    do.call(rbind, history)
  } else {
    data.frame(step = integer(0), action = character(0), variable = character(0),
               F = numeric(0), p_value = numeric(0), wilks = numeric(0),
               stringsAsFactors = FALSE)
  }
  list(selected = vars[vars %in% in_model], history = hist_df)
}
