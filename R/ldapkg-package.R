#' ldapkg: Predictive Linear Discriminant Analysis
#'
#' Predictive linear discriminant analysis with an R6 interface modelled on
#' scikit-learn (`fit()`, `predict()`, `predict_proba()`), statistical
#' diagnostics (Wilks' lambda, partial F statistics, canonical axes),
#' dummy coding of categorical predictors, stepwise variable selection and a
#' Shiny application.
#'
#' @section Main entry points:
#' * [LDAClassifier]: the R6 class implementing the model.
#' * [run_app()]: launches the Shiny application.
#' * [ldapkg_example()]: example datasets shipped with the package.
#'
#' @keywords internal
#' @importFrom graphics abline barplot boxplot legend mosaicplot par plot points text
#' @importFrom grDevices adjustcolor hcl.colors
"_PACKAGE"

#' @importFrom R6 R6Class
NULL
