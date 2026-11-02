X <- iris[, 1:4]
y <- iris$Species

test_that("posteriors, predictions and axes match MASS::lda", {
  skip_if_not_installed("MASS")
  m <- LDAClassifier$new()$fit(X, y)
  ref <- MASS::lda(X, grouping = y)
  pr <- predict(ref, X)
  expect_equal(m$predict_proba(X), pr$posterior, tolerance = 1e-8, ignore_attr = TRUE)
  expect_identical(as.character(m$predict(X)), as.character(pr$class))
  # discriminant axes are unique up to their sign
  expect_equal(abs(unname(m$axes)), abs(unname(ref$scaling)), tolerance = 1e-6)
  expect_equal(unname(m$means), unname(ref$means), tolerance = 1e-10)
})

test_that("equal priors and custom priors match MASS::lda", {
  skip_if_not_installed("MASS")
  m <- LDAClassifier$new(priors = "equal")$fit(X, y)
  ref <- MASS::lda(X, grouping = y, prior = rep(1 / 3, 3))
  expect_equal(m$predict_proba(X), predict(ref, X)$posterior, tolerance = 1e-8, ignore_attr = TRUE)
  pri <- c(setosa = 0.2, versicolor = 0.3, virginica = 0.5)
  m2 <- LDAClassifier$new(priors = pri)$fit(X, y)
  ref2 <- MASS::lda(X, grouping = y, prior = unname(pri))
  expect_equal(m2$predict_proba(X), predict(ref2, X)$posterior, tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("probabilities are proper and the model works with two classes", {
  skip_if_not_installed("MASS")
  i2 <- iris$Species != "setosa"
  X2 <- iris[i2, 1:4]; y2 <- droplevels(iris$Species[i2])
  m <- LDAClassifier$new()$fit(X2, y2)
  P <- m$predict_proba(X2)
  expect_equal(unname(rowSums(P)), rep(1, nrow(P)), tolerance = 1e-12)
  expect_true(all(P >= 0 & P <= 1))
  expect_equal(ncol(m$axes), 1L)
  ref <- MASS::lda(X2, grouping = y2)
  expect_equal(P, predict(ref, X2)$posterior, tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("Wilks lambda and Rao F match stats::manova", {
  m <- LDAClassifier$new()$fit(X, y)
  s <- summary(stats::manova(as.matrix(X) ~ y), test = "Wilks")$stats
  expect_equal(unname(m$wilks_test[["wilks"]]), unname(s[1, "Wilks"]), tolerance = 1e-8)
  expect_equal(unname(m$wilks_test[["F"]]), unname(s[1, "approx F"]), tolerance = 1e-6)
  expect_equal(unname(m$wilks_test[["df1"]]), unname(s[1, "num Df"]))
  expect_equal(unname(m$wilks_test[["df2"]]), unname(s[1, "den Df"]), tolerance = 1e-8)
  # Wilks = prod(1 / (1 + eigenvalues))
  expect_equal(m$wilks, prod(1 / (1 + m$eigenvalues)), tolerance = 1e-10)
})

# brute-force Wilks' lambda det(E) / det(T) with base R only
wilks_brute <- function(Z, g) {
  Z <- as.matrix(Z)
  Ec <- Z - apply(Z, 2, function(v) stats::ave(v, g))
  Tc <- scale(Z, scale = FALSE)
  det(crossprod(Ec)) / det(crossprod(Tc))
}

test_that("partial Wilks lambda matches a brute-force computation", {
  m <- LDAClassifier$new()$fit(X, y)
  vi <- m$var_importance
  full <- wilks_brute(X, y)
  expect_equal(m$wilks, full, tolerance = 1e-10)
  for (v in names(X)) {
    lam <- full / wilks_brute(X[, setdiff(names(X), v), drop = FALSE], y)
    expect_equal(vi$partial_wilks[vi$variable == v], lam, tolerance = 1e-8)
  }
})

test_that("training accuracy and confusion matrix are consistent", {
  m <- LDAClassifier$new()$fit(X, y)
  expect_equal(sum(m$confusion), nrow(X))
  expect_equal(m$accuracy, mean(m$predict(X) == y))
})

test_that("canonical scores are centred with unit within-class variance", {
  m <- LDAClassifier$new()$fit(X, y)
  expect_equal(unname(colMeans(m$scores)), rep(0, ncol(m$scores)), tolerance = 1e-8)
  Wc <- crossprod(m$scores - m$centroids[as.integer(y), ]) / (nrow(X) - 3)
  expect_equal(unname(Wc), diag(ncol(m$scores)), tolerance = 1e-8)
  expect_equal(m$transform(X), m$scores, tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("a larger problem (wine, 3 classes, 13 variables) agrees with MASS::lda", {
  skip_if_not_installed("MASS")
  w <- ldapkg_example("wine")
  Xw <- w[, setdiff(names(w), "cultivar")]
  m <- LDAClassifier$new()$fit(Xw, w$cultivar)
  ref <- MASS::lda(Xw, grouping = w$cultivar)
  expect_equal(m$predict_proba(Xw), predict(ref, Xw)$posterior, tolerance = 1e-7, ignore_attr = TRUE)
})
