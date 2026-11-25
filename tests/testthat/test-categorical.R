d <- ldapkg_example("customers")
d <- d[stats::complete.cases(d), ]
Xc <- d[, setdiff(names(d), "churn")]

test_that("dummy coding gives the same model as MASS::lda on model.matrix", {
  skip_if_not_installed("MASS")
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  Xd <- stats::model.matrix(~ ., data = Xc)[, -1, drop = FALSE]
  ref <- MASS::lda(Xd, grouping = d$churn)
  expect_equal(m$predict_proba(Xc), predict(ref, Xd)$posterior, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("categorical variables are tested as blocks", {
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  vi <- m$var_importance
  expect_setequal(vi$variable, names(Xc))
  # contract has 3 levels -> 2 dummy columns, region 4 levels -> 3
  expect_equal(vi$n_columns[vi$variable == "contract"], 2)
  expect_equal(vi$n_columns[vi$variable == "region"], 3)
  expect_equal(vi$df1[vi$variable == "region"], 3 * (2 - 1))
  expect_true(all(vi$p_value >= 0 & vi$p_value <= 1))
})

test_that("character and logical columns are supported", {
  Xch <- Xc
  Xch$region <- as.character(Xch$region)
  Xch$flag <- Xch$age > 45
  m <- LDAClassifier$new()$fit(Xch, as.character(d$churn))
  expect_s3_class(m$predict(Xch), "factor")
  expect_equal(m$classes, c("No", "Yes"))
})

test_that("describe_inputs lists the variables needed for deployment", {
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  info <- m$describe_inputs()
  nm <- vapply(info, function(s) s$name, character(1))
  expect_setequal(nm, names(Xc))
  tp <- vapply(info, function(s) s$type, character(1))
  expect_equal(unname(tp[nm == "region"]), "factor")
})

# ---- block statistics checked against a brute-force computation -----------

wilks_brute <- function(Z, g) {
  Z <- as.matrix(Z)
  if (ncol(Z) == 0L) return(1)
  Ec <- Z - apply(Z, 2, function(v) stats::ave(v, g))
  Tc <- scale(Z, scale = FALSE)
  det(crossprod(Ec)) / det(crossprod(Tc))
}

test_that("partial Wilks lambda of a categorical block matches brute force", {
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  Xd <- stats::model.matrix(~ ., data = Xc)[, -1, drop = FALSE]
  owner <- attr(stats::model.matrix(~ ., data = Xc), "assign")[-1]
  full <- wilks_brute(Xd, d$churn)
  vi <- m$var_importance
  for (j in seq_along(Xc)) {
    v <- names(Xc)[j]
    lam <- full / wilks_brute(Xd[, owner != j, drop = FALSE], d$churn)
    expect_equal(vi$partial_wilks[vi$variable == v], lam, tolerance = 1e-8)
  }
})

test_that("with two classes the block F is the exact F test", {
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  vi <- m$var_importance
  r <- vi[vi$variable == "region", ]
  # K = 2: Rao's F is exact (Hotelling), F = (1 - L) / L * df2 / df1 with
  # df1 = m and df2 = (n - K - (p - m)) - m + 1 = n - K - p + 1
  p <- length(m$features)
  expect_equal(r$df1, r$n_columns)
  expect_equal(r$df2, m$n_obs - 2 - p + 1)
  expect_equal(r$F, (1 - r$partial_wilks) / r$partial_wilks * r$df2 / r$df1,
               tolerance = 1e-10)
  expect_equal(r$p_value, stats::pf(r$F, r$df1, r$df2, lower.tail = FALSE),
               tolerance = 1e-10)
})

test_that("the choice of the reference level does not change the model", {
  X2 <- Xc
  X2$region <- stats::relevel(X2$region, ref = "West")
  m1 <- LDAClassifier$new()$fit(Xc, d$churn)
  m2 <- LDAClassifier$new()$fit(X2, d$churn)
  expect_equal(m1$predict_proba(Xc), m2$predict_proba(X2), tolerance = 1e-8)
  expect_equal(m1$var_importance, m2$var_importance, tolerance = 1e-8)
  expect_true("region=East" %in% m2$features)
  expect_false("region=West" %in% m2$features)
})

test_that("unused factor levels and single-level factors create no column", {
  X3 <- Xc
  X3$region <- factor(X3$region, levels = c(levels(X3$region), "Unused"))
  X3$one <- factor(rep("only", nrow(X3)))
  m <- LDAClassifier$new()$fit(X3, d$churn)
  expect_false(any(grepl("Unused", m$features)))
  expect_false("one" %in% m$variables)
  m0 <- LDAClassifier$new()$fit(Xc, d$churn)
  expect_equal(m$predict_proba(X3), m0$predict_proba(Xc), tolerance = 1e-8)
})

test_that("a model with only categorical predictors works", {
  Xf <- Xc[, c("region", "contract", "payment", "has_options")]
  m <- LDAClassifier$new()$fit(Xf, d$churn)
  expect_equal(nrow(m$var_importance), 4L)
  expect_equal(sum(m$var_importance$n_columns), length(m$features))
  expect_true(all(m$var_importance$tolerance > 0 & m$var_importance$tolerance <= 1))
})

test_that("the tolerance of a block is the generalised tolerance", {
  m <- LDAClassifier$new()$fit(Xc, d$churn)
  vi <- m$var_importance
  Xd <- stats::model.matrix(~ ., data = Xc)[, -1, drop = FALSE]
  owner <- attr(stats::model.matrix(~ ., data = Xc), "assign")[-1]
  Ec <- Xd - apply(Xd, 2, function(v) stats::ave(v, d$churn))   # within-class centring
  E <- crossprod(Ec)
  for (j in seq_along(Xc)) {
    v <- owner == j
    # det(E_vv | others) / det(E_vv) = det(E) / (det(E_oo) det(E_vv))
    gt <- det(E) / (det(E[!v, !v, drop = FALSE]) * det(E[v, v, drop = FALSE]))
    expect_equal(vi$tolerance[vi$variable == names(Xc)[j]], gt, tolerance = 1e-8)
  }
})
