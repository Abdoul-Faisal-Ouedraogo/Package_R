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
