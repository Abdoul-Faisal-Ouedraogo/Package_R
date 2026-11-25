test_that("cross-validation returns sensible, reproducible results", {
  X <- iris[, 1:4]; y <- iris$Species
  m <- LDAClassifier$new()$fit(X, y)
  r1 <- m$cv(X, y, folds = 5, seed = 1)
  r2 <- m$cv(X, y, folds = 5, seed = 1)
  expect_equal(r1$accuracy, r2$accuracy)
  expect_gt(r1$accuracy, 0.9)
  expect_equal(sum(r1$confusion), 150)
  expect_equal(m$cv_results$folds, 5L)
  expect_error(m$cv(X, y, folds = 1), "folds")
})

test_that("cross-validation re-runs variable selection inside each fold", {
  w <- ldapkg_example("wine")
  Xw <- w[, setdiff(names(w), "cultivar")]
  m <- LDAClassifier$new(selection = "forward", alpha_enter = 0.01)$fit(Xw, w$cultivar)
  r <- m$cv(Xw, w$cultivar, folds = 5, seed = 2)
  expect_gt(r$accuracy, 0.9)
  expect_equal(nrow(r$per_class), 3L)
})

test_that("cv(seed = ) does not change the random number stream of the session", {
  X <- iris[, 1:4]; y <- iris$Species
  m <- LDAClassifier$new()$fit(X, y)
  set.seed(99); a <- stats::runif(3)
  set.seed(99); invisible(m$cv(X, y, folds = 3, seed = 1)); b <- stats::runif(3)
  expect_equal(a, b)
})

test_that("cross-validation folds are stratified and results are consistent", {
  X <- iris[, 1:4]; y <- iris$Species
  r <- LDAClassifier$new()$cv(X, y, folds = 5, seed = 3)
  expect_length(r$fold_accuracy, 5L)
  expect_true(all(r$fold_accuracy >= 0 & r$fold_accuracy <= 1))
  expect_true(is.finite(r$accuracy_se) && r$accuracy_se >= 0)
  expect_equal(r$accuracy, sum(diag(r$confusion)) / sum(r$confusion))
  expect_equal(r$global[["accuracy"]], r$accuracy)
  expect_equal(r$per_class, .class_metrics(r$confusion))
  expect_equal(r$n_unclassified, 0L)
  expect_null(r$selection_frequency)
  # stratification: equal class sizes and 5 folds -> every fold has 10 per class,
  # so the mean of the fold accuracies equals the global accuracy
  expect_equal(mean(r$fold_accuracy), r$accuracy)
})

test_that("cross-validation reports how often each variable is selected", {
  w <- ldapkg_example("wine")
  Xw <- w[, setdiff(names(w), "cultivar")]
  m <- LDAClassifier$new(selection = "stepwise")
  r <- m$cv(Xw, w$cultivar, folds = 5, seed = 1)
  f <- r$selection_frequency
  expect_setequal(names(f), names(Xw))
  expect_true(all(f >= 0 & f <= 1))
  expect_false(is.unsorted(rev(f)))
  expect_equal(max(f), 1)          # the strongest variables are always kept
})

test_that("held-out rows with a level unseen in training are counted, not hidden", {
  d <- ldapkg_example("customers")
  d <- d[stats::complete.cases(d), ]
  Xc <- d[, setdiff(names(d), "churn")]
  Xc$region <- as.character(Xc$region)
  Xc$region[1] <- "Rare"
  m <- LDAClassifier$new()
  expect_warning(r <- m$cv(Xc, d$churn, folds = 5, seed = 1), "could not be classified")
  expect_equal(r$n_unclassified, 1L)
  expect_equal(r$n + r$n_unclassified, nrow(Xc))
})

test_that("equal priors improve the balanced accuracy on unbalanced classes", {
  d <- ldapkg_example("customers")
  d <- d[stats::complete.cases(d), ]
  Xc <- d[, setdiff(names(d), "churn")]
  r_emp <- LDAClassifier$new()$cv(Xc, d$churn, folds = 5, seed = 1)
  r_eq <- LDAClassifier$new(priors = "equal")$cv(Xc, d$churn, folds = 5, seed = 1)
  expect_gt(r_eq$global[["balanced_accuracy"]], r_emp$global[["balanced_accuracy"]])
  expect_gt(r_emp$accuracy, r_eq$accuracy)   # ... at the cost of raw accuracy
})

test_that("summary reports global metrics and cross-validation details", {
  w <- ldapkg_example("wine")
  Xw <- w[, setdiff(names(w), "cultivar")]
  m <- LDAClassifier$new(selection = "forward")$fit(Xw, w$cultivar)
  m$cv(Xw, w$cultivar, folds = 5, seed = 1)
  out <- utils::capture.output(res <- m$summary())
  expect_true(any(grepl("Balanced accuracy", out)))
  expect_true(any(grepl("Kappa", out)))
  expect_true(any(grepl("Proportion of folds", out)))
  expect_true(any(grepl("s.e.", out, fixed = TRUE)))
  expect_named(res$global_metrics, names(m$global_metrics()))
})
