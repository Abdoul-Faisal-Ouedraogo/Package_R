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
