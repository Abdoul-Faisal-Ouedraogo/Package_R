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
