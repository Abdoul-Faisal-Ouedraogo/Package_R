#' Example datasets shipped with the package
#'
#' Tab-separated files stored in `inst/extdata`.
#'
#' @details
#' * `"wine"`: 178 wines, 13 numeric chemical measurements, target `cultivar`
#'   (3 classes). UCI Wine data.
#' * `"breast_cancer"`: 569 tumours, 30 numeric (strongly correlated) image
#'   features, target `diagnosis` (malignant / benign). UCI Wisconsin
#'   Diagnostic Breast Cancer data.
#' * `"customers"`: 1500 **synthetic** customers with numeric (`age`, `income`,
#'   `tenure_months`, `monthly_charges`) and categorical (`region`, `contract`,
#'   `payment`, `has_options`) predictors, a few missing values in `income`,
#'   target `churn` (Yes / No).
#'
#' @param name Name of a dataset; `NULL` (default) returns the available names.
#' @param path Logical. If `TRUE`, return the path of the file instead of the
#'   data (useful to try the file upload of the Shiny application).
#' @return A character vector of names, a data frame (character columns are
#'   converted to factors) or a file path.
#' @examples
#' ldapkg_example()
#' wine <- ldapkg_example("wine")
#' str(wine)
#' ldapkg_example("wine", path = TRUE)
#' @export
ldapkg_example <- function(name = NULL, path = FALSE) {
  dir <- system.file("extdata", package = "ldapkg")
  avail <- sub("\\.tsv$", "", list.files(dir, pattern = "\\.tsv$"))
  if (is.null(name)) return(avail)
  name <- match.arg(name, avail)
  f <- file.path(dir, paste0(name, ".tsv"))
  if (isTRUE(path)) return(f)
  utils::read.delim(f, stringsAsFactors = TRUE, na.strings = "NA")
}
