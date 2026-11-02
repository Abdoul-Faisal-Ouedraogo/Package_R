#' Launch the Shiny application
#'
#' Opens the interactive application: load a file (csv / tsv / txt / xlsx) or
#' an example dataset, choose the target, explanatory and illustrative
#' variables, fit the model (with optional variable selection, hold-out
#' sample and cross-validation), explore the results and deploy the model on
#' new individuals (one by one or from a file).
#'
#' The application needs the suggested packages `shiny` and `DT`; `readxl`
#' (Excel files) and `data.table` (fast reading of large files) are optional.
#'
#' @param ... Arguments passed to [shiny::runApp()] (e.g. `port`, `launch.browser`).
#' @return Called for its side effect (does not return while the app runs).
#' @examples
#' if (interactive()) run_app()
#' @export
run_app <- function(...) {
  for (pkg in c("shiny", "DT")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package '", pkg, "' is required to run the application. ",
           "Install it with install.packages(\"", pkg, "\").", call. = FALSE)
    }
  }
  app_dir <- system.file("shiny", "ldaapp", package = "ldapkg")
  if (!nzchar(app_dir)) {
    stop("Could not find the application directory. Try re-installing 'ldapkg'.",
         call. = FALSE)
  }
  shiny::runApp(app_dir, ...)
}
