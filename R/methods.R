# S3 wrappers so that the usual generics work on LDAClassifier objects ----

#' Predict method for LDAClassifier objects
#'
#' @param object A fitted [LDAClassifier].
#' @param newdata A data frame of explanatory variables.
#' @param type `"class"` (default) returns the predicted classes, `"prob"` the
#'   matrix of posterior probabilities.
#' @param ... Ignored.
#' @return A factor (`type = "class"`) or a numeric matrix (`type = "prob"`).
#' @examples
#' m <- LDAClassifier$new()$fit(iris[, 1:4], iris$Species)
#' predict(m, iris[c(1, 51, 101), 1:4])
#' predict(m, iris[c(1, 51, 101), 1:4], type = "prob")
#' @export
predict.LDAClassifier <- function(object, newdata, type = c("class", "prob"), ...) {
  type <- match.arg(type)
  if (type == "class") object$predict(newdata) else object$predict_proba(newdata)
}

#' Summary method for LDAClassifier objects
#'
#' Calls the `$summary()` method of the object.
#' @param object A fitted [LDAClassifier].
#' @param ... Passed to `$summary()` (e.g. `digits`).
#' @return Invisibly, the list returned by `$summary()`.
#' @examples
#' m <- LDAClassifier$new()$fit(iris[, 1:4], iris$Species)
#' summary(m)
#' @export
summary.LDAClassifier <- function(object, ...) {
  object$summary(...)
}

#' Plot method for LDAClassifier objects
#'
#' Calls the `$plot()` method of the object.
#' @param x A fitted [LDAClassifier].
#' @param ... Passed to `$plot()` (`type`, `dims`, `max_points`).
#' @return The object, invisibly.
#' @examples
#' m <- LDAClassifier$new()$fit(iris[, 1:4], iris$Species)
#' plot(m)
#' plot(m, type = "importance")
#' @export
plot.LDAClassifier <- function(x, ...) {
  x$plot(...)
}
