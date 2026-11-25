# Evaluation metrics (R/metrics.R) ------------------------------------------

# 3 classes, observed in rows, predicted in columns
cm3 <- matrix(c(8, 1, 1,
                2, 6, 2,
                0, 3, 7), 3, byrow = TRUE,
              dimnames = list(Observed = c("a", "b", "c"),
                              Predicted = c("a", "b", "c")))

test_that("per-class metrics match a hand computation", {
  pc <- .class_metrics(cm3)
  expect_equal(pc$class, c("a", "b", "c"))
  expect_equal(pc$n, c(10L, 10L, 10L))
  expect_equal(pc$recall, c(8, 6, 7) / 10)
  expect_equal(pc$precision, c(8 / 10, 6 / 10, 7 / 10))
  r <- pc$recall; p <- pc$precision
  expect_equal(pc$f1, 2 * r * p / (r + p))
})

test_that("global metrics match their definitions", {
  g <- .global_metrics(cm3)
  n <- sum(cm3)
  po <- sum(diag(cm3)) / n
  pe <- sum(rowSums(cm3) * colSums(cm3)) / n^2
  expect_equal(g[["n"]], 30)
  expect_equal(g[["accuracy"]], 21 / 30)
  expect_equal(g[["error_rate"]], 9 / 30)
  expect_equal(g[["balanced_accuracy"]], mean(c(0.8, 0.6, 0.7)))
  expect_equal(g[["kappa"]], (po - pe) / (1 - pe))
  pc <- .class_metrics(cm3)
  expect_equal(g[["macro_f1"]], mean(pc$f1))
  expect_equal(g[["weighted_f1"]], sum(pc$n * pc$f1) / n)
})

test_that("balanced accuracy differs from accuracy with unequal classes", {
  cm <- matrix(c(90, 10, 0, 0), 2, dimnames = list(c("No", "Yes"), c("No", "Yes")))
  # everything predicted "No": 90 % accuracy but only 50 % balanced accuracy
  g <- .global_metrics(cm)
  expect_equal(g[["accuracy"]], 0.9)
  expect_equal(g[["balanced_accuracy"]], 0.5)
  expect_equal(g[["kappa"]], 0)
  pc <- .class_metrics(cm)
  expect_true(is.na(pc$precision[pc$class == "Yes"]))   # never predicted
  expect_true(is.na(pc$f1[pc$class == "Yes"]))
  expect_equal(g[["macro_f1"]], mean(c(2 * 0.9 / 1.9, 0)))  # undefined F1 counts 0
})

test_that("a perfect classification gives 1 everywhere", {
  cm <- diag(c(5, 7, 3))
  dimnames(cm) <- list(c("a", "b", "c"), c("a", "b", "c"))
  g <- .global_metrics(cm)
  expect_equal(unname(g[c("accuracy", "balanced_accuracy", "macro_f1", "kappa")]),
               rep(1, 4))
})

test_that("confusion tables with missing or reordered columns are aligned", {
  y <- factor(c("a", "b", "c", "c"))
  pred <- factor(c("a", "a", "c", "c"), levels = c("c", "a"))   # 'b' never predicted
  tab <- table(Observed = y, Predicted = pred)
  expect_equal(dim(tab), c(3L, 2L))
  pc <- .class_metrics(tab)
  expect_equal(pc$recall, c(1, 0, 1))
  expect_equal(pc$precision, c(0.5, NA, 1))
  expect_equal(.global_metrics(tab)[["accuracy"]], 3 / 4)
})

test_that("invalid confusion matrices give clear errors", {
  expect_error(.class_metrics(1:4), "two-dimensional")
  expect_error(.class_metrics(matrix(1:6, 2)), "square")
  expect_error(.class_metrics(matrix(c(1, -1, 0, 2), 2)), "non-negative")
  expect_error(.global_metrics(matrix(0, 2, 2)), "empty")
})

test_that("the R6 methods expose the metrics", {
  m <- LDAClassifier$new()$fit(iris[, 1:4], iris$Species)
  expect_equal(m$class_metrics(), .class_metrics(m$confusion))
  g <- m$global_metrics()
  expect_equal(g[["accuracy"]], m$accuracy)
  expect_named(g, c("n", "accuracy", "error_rate", "balanced_accuracy",
                    "macro_precision", "macro_recall", "macro_f1",
                    "weighted_f1", "kappa"))
  expect_equal(m$global_metrics(cm3)[["accuracy"]], 21 / 30)
  expect_error(LDAClassifier$new()$global_metrics(), "not been fitted")
})
