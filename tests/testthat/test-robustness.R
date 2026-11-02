X <- iris[, 1:4]
y <- iris$Species
m <- LDAClassifier$new()$fit(X, y)

test_that("predict ignores column order and extra columns", {
  Xs <- X[, 4:1]
  Xs$extra <- "ignored"
  expect_identical(m$predict(Xs), m$predict(X))
})

test_that("missing columns, wrong inputs and unfitted models raise clear errors", {
  expect_error(m$predict(X[, 1:3]), "Missing column")
  expect_error(m$predict(1:5), "data.frame")
  expect_error(LDAClassifier$new()$predict(X), "not been fitted")
  expect_error(LDAClassifier$new()$fit(X, y[-1]), "same number")
  expect_error(LDAClassifier$new()$fit(X, 1:150), "factor")
  expect_error(LDAClassifier$new()$fit(X, factor(rep("a", 150))), "two")
  expect_error(LDAClassifier$new(priors = "bad"), "priors")
  expect_error(LDAClassifier$new(priors = c(a = 1, b = 1))$fit(X, y), "priors")
  expect_error(LDAClassifier$new(alpha_enter = 2), "alpha")
})

test_that("rows with missing values are handled", {
  Xn <- X[1:6, ]
  Xn[2, 1] <- NA
  p <- m$predict(Xn)
  expect_true(is.na(p[2]))
  expect_false(anyNA(p[-2]))
  P <- m$predict_proba(Xn)
  expect_true(all(is.na(P[2, ])))
  # at fit time
  Xf <- X; Xf[3, 2] <- NA
  expect_message(mf <- LDAClassifier$new()$fit(Xf, y), "removed")
  expect_equal(mf$n_obs, 149L)
  expect_false(3L %in% mf$rows_used)
  expect_error(LDAClassifier$new(na_action = "fail")$fit(Xf, y), "Missing")
})

test_that("constant and collinear columns are dropped with a warning", {
  X2 <- X
  X2$const <- 1
  X2$dup <- 2 * X2$Sepal.Length + X2$Petal.Width
  expect_warning(m2 <- LDAClassifier$new()$fit(X2, y), "Dropped")
  expect_true(all(c("const", "dup") %in% m2$dropped))
  expect_equal(m2$predict_proba(X2), m$predict_proba(X), tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("p larger than n - K does not crash", {
  set.seed(1)
  Xw <- as.data.frame(matrix(rnorm(20 * 30), 20, 30))
  yw <- factor(rep(c("a", "b"), each = 10))
  expect_warning(mw <- LDAClassifier$new()$fit(Xw, yw), "Dropped")
  expect_s3_class(mw$predict(Xw), "factor")
})

test_that("unseen factor levels give NA with a warning", {
  d <- ldapkg_example("customers")
  Xc <- d[, c("age", "region", "contract")]
  mc <- suppressMessages(LDAClassifier$new()$fit(Xc, d$churn))
  nd <- Xc[1:3, ]
  nd$region <- as.character(nd$region)
  nd$region[2] <- "Atlantis"
  expect_warning(p <- mc$predict(nd), "not seen")
  expect_true(is.na(p[2]))
  expect_false(anyNA(p[-2]))
})

test_that("completely unclassifiable rows remain NA", {
  Xn <- X[1:3, , drop = FALSE]
  Xn[2, ] <- NA
  expect_true(is.na(m$predict(Xn)[2]))
  expect_true(all(is.na(m$predict_proba(Xn)[2, ])))
})

test_that("cross-validation protects the class representation", {
  Xsmall <- X[c(1:3, 51:53, 101:103), , drop = FALSE]
  ysmall <- droplevels(y[c(1:3, 51:53, 101:103)])
  expect_warning(r <- m$cv(Xsmall, ysmall, folds = 10, seed = 1), "Reducing")
  expect_equal(r$folds, 3L)
  yone <- factor(c("a", "a", "b"))
  expect_error(LDAClassifier$new()$cv(Xsmall[1:3, ], yone, folds = 2),
               "at least two observations")
})

test_that("S3 wrappers, print and summary work", {
  expect_output(print(m), "LDAClassifier")
  expect_output(summary(m), "Wilks")
  expect_identical(predict(m, X), m$predict(X))
  expect_equal(predict(m, X, type = "prob"), m$predict_proba(X))
  expect_output(LDAClassifier$new()$print(), "not fitted")
  s <- suppressWarnings(utils::capture.output(res <- summary(m)))
  expect_true(all(c("var_importance", "confusion", "axes") %in% names(res)))
})

test_that("fit() can be chained and a model can be refitted", {
  m3 <- LDAClassifier$new()
  expect_identical(m3$fit(X, y), m3)
  m3$fit(X[, 1:2], y)
  expect_equal(length(m3$variables), 2L)
})

test_that("plot validates user supplied dimensions", {
  expect_error(m$plot("axes", dims = c(1, 1)), "two distinct")
  expect_error(m$plot("axes", dims = c(1, 99)), "two distinct")
  expect_error(m$plot("importance", max_points = 0), "positive")
})
