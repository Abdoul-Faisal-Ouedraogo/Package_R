w <- ldapkg_example("wine")
Xw <- w[, setdiff(names(w), "cultivar")]
yw <- w$cultivar

test_that("the first forward step picks the variable with the smallest Wilks lambda", {
  m <- LDAClassifier$new(selection = "forward", alpha_enter = 0.01)$fit(Xw, yw)
  uni <- vapply(Xw, function(v) sum((v - stats::ave(v, yw))^2) / sum((v - mean(v))^2), numeric(1))
  expect_equal(m$selection_history$variable[1], names(which.min(uni)))
  expect_true(length(m$variables) >= 1L && length(m$variables) <= ncol(Xw))
  expect_true(all(m$selection_history$p_value[m$selection_history$action == "enter"] < 0.01))
})

test_that("backward elimination stops when every remaining variable is significant", {
  m <- LDAClassifier$new(selection = "backward", alpha_stay = 0.05)$fit(Xw, yw)
  expect_true(all(m$var_importance$p_value <= 0.05 + 1e-12))
  expect_true(all(m$selection_history$action == "remove"))
})

test_that("stepwise selection runs and the final model only needs the selected variables", {
  m <- LDAClassifier$new(selection = "stepwise", alpha_enter = 0.05, alpha_stay = 0.10)$fit(Xw, yw)
  expect_true(length(m$variables) < ncol(Xw))
  expect_s3_class(m$predict(Xw[, m$variables, drop = FALSE]), "factor")
  # the reduced model equals an unselected model fitted on the same variables
  m0 <- LDAClassifier$new()$fit(Xw[, m$variables, drop = FALSE], yw)
  expect_equal(m$predict_proba(Xw), m0$predict_proba(Xw), tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("selection treats categorical variables as blocks", {
  d <- ldapkg_example("customers")
  d <- d[stats::complete.cases(d), ]
  Xc <- d[, setdiff(names(d), "churn")]
  m <- LDAClassifier$new(selection = "stepwise")$fit(Xc, d$churn)
  expect_true(all(m$variables %in% names(Xc)))
  # each selected factor keeps all its dummy columns
  for (v in intersect(m$variables, c("region", "contract", "payment", "has_options"))) {
    nlev <- nlevels(Xc[[v]])
    expect_equal(sum(startsWith(m$features, paste0(v, "="))), nlev - 1L)
  }
})

test_that("a very strict threshold still returns a usable model", {
  expect_warning(m <- LDAClassifier$new(selection = "forward", alpha_enter = 1e-300)$fit(Xw, yw),
                 "single best")
  expect_equal(length(m$variables), 1L)
})

test_that("collinear data (breast cancer, 30 variables) can be fitted and selected", {
  bc <- ldapkg_example("breast_cancer")
  Xb <- bc[, setdiff(names(bc), "diagnosis")]
  m <- LDAClassifier$new(selection = "forward", alpha_enter = 0.01)$fit(Xb, bc$diagnosis)
  expect_gt(m$accuracy, 0.9)
  expect_lt(length(m$variables), ncol(Xb))
})

# ---- selection statistics checked against brute force ---------------------

wilks_brute <- function(Z, g) {
  Z <- as.matrix(Z)
  if (ncol(Z) == 0L) return(1)
  Ec <- Z - apply(Z, 2, function(v) stats::ave(v, g))
  Tc <- scale(Z, scale = FALSE)
  det(crossprod(Ec)) / det(crossprod(Tc))
}

test_that("F to enter and the Wilks lambda of each step match brute force", {
  m <- LDAClassifier$new(selection = "forward", alpha_enter = 0.01)$fit(Xw, yw)
  h <- m$selection_history
  n <- nrow(Xw); K <- nlevels(yw)
  for (s in seq_len(nrow(h))) {
    S <- h$variable[seq_len(s)]
    prev <- h$variable[seq_len(s - 1L)]
    expect_equal(h$wilks[s], wilks_brute(Xw[, S, drop = FALSE], yw), tolerance = 1e-8)
    lam <- wilks_brute(Xw[, S, drop = FALSE], yw) / wilks_brute(Xw[, prev, drop = FALSE], yw)
    rf <- .rao_F(lam, 1, K - 1, n - K - (s - 1))
    expect_equal(h$F[s], rf[["F"]], tolerance = 1e-6)
  }
})

test_that("each forward step picks the best candidate", {
  m <- LDAClassifier$new(selection = "forward", alpha_enter = 0.01)$fit(Xw, yw)
  h <- m$selection_history
  for (s in seq_len(nrow(h))) {
    prev <- h$variable[seq_len(s - 1L)]
    cand <- setdiff(names(Xw), prev)
    lam <- vapply(cand, function(v) wilks_brute(Xw[, c(prev, v), drop = FALSE], yw),
                  numeric(1))
    expect_equal(h$variable[s], names(which.min(lam)))
  }
})

test_that("selection is invariant to the units and the order of the columns", {
  m1 <- LDAClassifier$new(selection = "stepwise")$fit(Xw, yw)
  Xs <- Xw[, rev(names(Xw))]
  Xs[[1]] <- Xs[[1]] * 1e4
  Xs[[2]] <- Xs[[2]] / 1e3
  m2 <- LDAClassifier$new(selection = "stepwise")$fit(Xs, yw)
  expect_setequal(m1$variables, m2$variables)
  expect_equal(m1$selection_history$variable, m2$selection_history$variable)
  expect_equal(m1$predict_proba(Xw), m2$predict_proba(Xs), tolerance = 1e-8)
})

test_that("a permissive backward selection keeps every variable", {
  m <- LDAClassifier$new(selection = "backward", alpha_stay = 1)$fit(Xw, yw)
  expect_equal(nrow(m$selection_history), 0L)
  expect_named(m$selection_history, c("step", "action", "variable", "F", "p_value", "wilks"))
  expect_setequal(m$variables, names(Xw))
})

test_that("pure noise variables are not selected", {
  set.seed(42)
  Xn <- Xw
  for (j in 1:5) Xn[[paste0("noise", j)]] <- stats::rnorm(nrow(Xw))
  m <- LDAClassifier$new(selection = "stepwise", alpha_enter = 0.01, alpha_stay = 0.01)$fit(Xn, yw)
  expect_false(any(startsWith(m$variables, "noise")))
})

test_that("forward, backward and stepwise agree on the customers data", {
  d <- ldapkg_example("customers")
  d <- d[stats::complete.cases(d), ]
  Xc <- d[, setdiff(names(d), "churn")]
  sel <- lapply(c("forward", "backward", "stepwise"), function(meth)
    sort(LDAClassifier$new(selection = meth)$fit(Xc, d$churn)$variables))
  expect_equal(sel[[1]], sel[[2]])
  expect_equal(sel[[1]], sel[[3]])
  expect_true("contract" %in% sel[[1]])
})
