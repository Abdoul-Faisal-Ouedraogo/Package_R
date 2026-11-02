#' Predictive Linear Discriminant Analysis
#'
#' @description
#' R6 class implementing predictive linear discriminant analysis (LDA) for a
#' categorical target with two or more classes. The interface follows
#' scikit-learn: create an object, call `$fit(X, y)`, then `$predict(X)` or
#' `$predict_proba(X)`.
#'
#' @details
#' ## Model
#' With class means \eqn{m_k}, pooled within-class covariance \eqn{W} (divisor
#' \eqn{n - K}) and prior probabilities \eqn{\pi_k}, the classification
#' function of class \eqn{k} is
#' \deqn{\delta_k(x) = x^\top W^{-1} m_k - \frac{1}{2} m_k^\top W^{-1} m_k + \log \pi_k.}
#' An observation is assigned to the class with the largest score and the
#' posterior probabilities are the softmax of the scores. Results are
#' identical to `MASS::lda()` for numeric predictors.
#'
#' ## Computation
#' Sufficient statistics (class means, within-class and between-class SSCP
#' matrices) are obtained in one pass with `rowsum()` and `crossprod()`;
#' everything else (coefficients, Wilks' lambda, partial statistics, axes) is
#' derived from these `p x p` matrices with Cholesky factorisations. Constant
#' and perfectly collinear columns are detected and dropped with a warning
#' instead of making the computation fail.
#'
#' ## Categorical predictors
#' Factor and character columns are replaced by `k - 1` dummy columns (the
#' first level is the reference; logical columns are used as 0/1); the encoding is
#' stored so that new data are encoded identically in `$predict()`. A
#' categorical variable is always handled as a block: in variable selection and
#' in the partial statistics, all its dummy columns enter or leave together and
#' the test has `(k - 1) * (K - 1)` degrees of freedom.
#'
#' ## Statistics
#' * **Global test**: Wilks' lambda \eqn{\Lambda = \det E / \det T} with Rao's
#'   F approximation (same as `summary(manova(), test = "Wilks")`).
#' * **Variable importance**: partial Wilks' lambda of each variable given the
#'   others, with the corresponding F ("F to remove") and p-value; the
#'   tolerance is one minus the squared multiple correlation of the variable
#'   with the others, within classes.
#' * **Discriminant axes**: eigenvectors of \eqn{E^{-1} B}, scaled so that the
#'   within-class variance is 1 (as in `MASS::lda()`), with eigenvalues and
#'   canonical correlations.
#' * **Selection**: `"forward"`, `"backward"` or `"stepwise"` on Wilks' lambda
#'   with entry / stay thresholds on the p-value of Rao's F.
#'
#' @section Reading the results:
#' After `$fit()`, `$coefficients` (one column per class) and `$intercept`
#' give the classification functions; the class with the highest score
#' \eqn{x^\top a_k + b_k} wins. A small `$wilks` (close to 0) and a small
#' p-value in `$wilks_test` indicate that the classes are well separated. In
#' `$var_importance`, a large `F` (small `p_value`) means that the variable
#' brings information beyond the other variables; `partial_wilks` close to 1
#' means a negligible contribution. `$confusion` is the resubstitution
#' confusion matrix (optimistic): use `$cv()` for an honest estimate.
#'
#' @field coefficients Matrix `p x K` of classification function coefficients
#'   (rows: encoded predictors, columns: classes).
#' @field intercept Named numeric vector of length `K`, constants of the
#'   classification functions (they include `log(priors)`).
#' @field classes Character vector of class labels.
#' @field priors Named numeric vector of prior probabilities.
#' @field class_counts Named integer vector, number of training observations per class.
#' @field means Matrix `K x p` of class means of the encoded predictors.
#' @field features Names of the encoded columns used by the model.
#' @field variables Names of the original variables used by the model
#'   (after variable selection).
#' @field dropped Names of the encoded columns removed because they were
#'   constant or collinear.
#' @field wilks Wilks' lambda of the model.
#' @field wilks_test Named numeric vector (`wilks`, `F`, `df1`, `df2`,
#'   `p_value`): Rao's F approximation of the global test.
#' @field var_importance Data frame with one row per variable: `partial_wilks`,
#'   `F`, `df1`, `df2`, `p_value`, `tolerance`.
#' @field axes Matrix `p x min(p, K - 1)` of canonical discriminant axes.
#' @field eigenvalues Eigenvalues of `E^-1 B` (one per axis).
#' @field canonical_cor Canonical correlations (one per axis).
#' @field scores Matrix of training observations on the discriminant axes.
#' @field centroids Matrix `K x r` of class centroids on the discriminant axes.
#' @field confusion Resubstitution confusion matrix (observed in rows).
#' @field accuracy Resubstitution accuracy.
#' @field cv_results List filled by `$cv()`.
#' @field selection_history Data frame describing the steps of variable
#'   selection (`NULL` when `selection = "none"`).
#' @field n_obs Number of observations used for fitting.
#' @field rows_used Indices of the rows of `X` used for fitting (rows with
#'   missing values are removed).
#'
#' @examples
#' X <- iris[, 1:4]
#' y <- iris$Species
#'
#' model <- LDAClassifier$new()
#' model$fit(X, y)
#' model                       # short description
#' model$coefficients          # classification functions
#' model$wilks_test            # global test
#' model$var_importance        # contribution of each variable
#'
#' new_obs <- X[c(1, 51, 101), ]
#' model$predict(new_obs)
#' round(model$predict_proba(new_obs), 3)
#'
#' summary(model)              # detailed report
#'
#' # categorical predictors + stepwise selection
#' cust <- ldapkg_example("customers")
#' m2 <- LDAClassifier$new(selection = "stepwise")
#' m2$fit(cust[, setdiff(names(cust), "churn")], cust$churn)
#' m2$variables
#' m2$selection_history
#'
#' \donttest{
#' model$cv(X, y, folds = 5, seed = 1)$accuracy
#' plot(model)
#' }
#' @export
LDAClassifier <- R6Class("LDAClassifier",
  public = list(
    coefficients = NULL,
    intercept = NULL,
    classes = NULL,
    priors = NULL,
    class_counts = NULL,
    means = NULL,
    features = NULL,
    variables = NULL,
    dropped = NULL,
    wilks = NA_real_,
    wilks_test = NULL,
    var_importance = NULL,
    axes = NULL,
    eigenvalues = NULL,
    canonical_cor = NULL,
    scores = NULL,
    centroids = NULL,
    confusion = NULL,
    accuracy = NA_real_,
    cv_results = NULL,
    selection_history = NULL,
    n_obs = NA_integer_,
    rows_used = NULL,

    #' @description Create a new (unfitted) model.
    #' @param priors Prior probabilities of the classes: `NULL` (default,
    #'   empirical class frequencies), `"equal"`, or a numeric vector with one
    #'   value per class (named by class, or in the order of the factor levels);
    #'   it is rescaled to sum to 1.
    #' @param selection Variable selection: `"none"` (default), `"forward"`,
    #'   `"backward"` or `"stepwise"`.
    #' @param alpha_enter Significance level (p-value of Rao's F) below which a
    #'   variable may enter the model (`"forward"`, `"stepwise"`).
    #' @param alpha_stay Significance level above which a variable is removed
    #'   (`"backward"`, `"stepwise"`). Should be at least `alpha_enter`.
    #' @param na_action `"omit"` (default) removes the rows with missing values
    #'   (with a message), `"fail"` raises an error.
    #' @param verbose Logical, print a message about the selected variables.
    #' @return A new `LDAClassifier` object.
    initialize = function(priors = NULL,
                          selection = c("none", "forward", "backward", "stepwise"),
                          alpha_enter = 0.05, alpha_stay = 0.05,
                          na_action = c("omit", "fail"), verbose = FALSE) {
      selection <- match.arg(selection)
      na_action <- match.arg(na_action)
      for (a in list(alpha_enter, alpha_stay)) {
        if (!is.numeric(a) || length(a) != 1L || is.na(a) || a <= 0 || a > 1) {
          stop("'alpha_enter' and 'alpha_stay' must be single numbers in (0, 1].",
               call. = FALSE)
        }
      }
      if (selection == "stepwise" && alpha_stay < alpha_enter) {
        warning("'alpha_stay' < 'alpha_enter': variables may enter and leave repeatedly.",
                call. = FALSE)
      }
      if (!is.null(priors)) {
        if (is.character(priors)) {
          if (!(length(priors) == 1L && priors == "equal")) {
            stop("'priors' must be NULL, \"equal\" or a numeric vector.", call. = FALSE)
          }
        } else if (!is.numeric(priors) || any(!is.finite(priors)) || any(priors <= 0)) {
          stop("'priors' must contain strictly positive numbers.", call. = FALSE)
        }
      }
      private$.params <- list(priors = priors, selection = selection,
                              alpha_enter = alpha_enter, alpha_stay = alpha_stay,
                              na_action = na_action, verbose = isTRUE(verbose))
      invisible(self)
    },

    #' @description Fit the model on training data.
    #' @param X A data frame of explanatory variables (numeric, logical,
    #'   factor or character columns; at least one column, unique names).
    #' @param y A factor (or character / logical vector) with the class of each
    #'   row of `X`; at least two non-empty classes.
    #' @return The object itself, invisibly (so calls can be chained).
    fit = function(X, y) {
      prm <- private$.params
      d <- private$prepare_xy(X, y)
      X <- d$X
      y <- d$y
      n <- nrow(X)
      K <- nlevels(y)
      if (n == 0L) {
        stop("'X' must contain at least one observation.", call. = FALSE)
      }
      if (K < 2L) stop("'y' must have at least two non-empty classes.", call. = FALSE)
      if (n <= K) stop("Not enough observations: n must exceed the number of classes.",
                       call. = FALSE)

      # 1. encoding (dummy coding of categorical predictors)
      spec <- .build_spec(X)
      enc <- .encode_matrix(X, spec)
      Xm <- enc$Xm
      assign <- enc$assign
      if (ncol(Xm) == 0L) stop("No usable predictor in 'X'.", call. = FALSE)
      if (!all(is.finite(Xm))) {
        stop("'X' contains infinite or non-numeric values.", call. = FALSE)
      }

      # 2. constant columns, then sufficient statistics and collinear columns
      const <- .constant_cols(Xm)
      dropped <- colnames(Xm)[const]
      if (all(const)) stop("All predictors are constant.", call. = FALSE)
      if (any(const)) {
        Xm <- Xm[, !const, drop = FALSE]
        assign <- assign[!const]
      }
      st <- .suff_stats(Xm, y)
      ind <- .independent_columns(st$E, st$T)
      if (length(ind) == 0L) {
        stop("No usable predictor: all columns are constant within classes.", call. = FALSE)
      }
      if (length(ind) < ncol(Xm)) {
        dropped <- c(dropped, colnames(Xm)[-ind])
        Xm <- Xm[, ind, drop = FALSE]
        assign <- assign[ind]
        st <- .subset_stats(st, ind)
      }
      if (length(dropped) > 0L) {
        warning("Dropped constant or collinear column(s): ",
                paste(dropped, collapse = ", "), call. = FALSE)
      }

      # 3. standardise (invariant for LDA, much better conditioned), priors and
      #    optional variable selection (on E and T only)
      st <- .scale_stats(st, sqrt(diag(st$E)))
      pri <- private$resolve_priors(st$cnt, levels(y))
      sel_hist <- NULL
      if (prm$selection != "none") {
        sel <- .select_variables(st$E, st$T, assign, n, K, prm$selection,
                                 prm$alpha_enter, prm$alpha_stay)
        keep <- which(assign %in% sel$selected)
        if (length(keep) < ncol(Xm)) {
          Xm <- Xm[, keep, drop = FALSE]
          assign <- assign[keep]
          st <- .subset_stats(st, keep)
        }
        sel_hist <- sel$history
        if (prm$verbose) {
          message("Selected variable(s): ", paste(sel$selected, collapse = ", "))
        }
      }

      # 4. solve
      sol <- .lda_solve(st, pri)
      vi <- .partial_table(sol$Einv, sol$Tinv, rep(1, ncol(Xm)), assign, n, K)

      # means and grand mean back in the original units
      means_o <- sweep(st$means, 2L, st$scale, "*")
      grand_o <- st$grand * st$scale

      # 5. resubstitution results and training scores on the axes
      S <- Xm %*% sol$coef
      S <- S + rep(sol$intercept, each = n)
      pred <- factor(levels(y)[max.col(S, ties.method = "first")], levels = levels(y))
      conf <- table(Observed = y, Predicted = pred)
      sc <- Xm %*% sol$axes
      sc <- sc - rep(drop(crossprod(grand_o, sol$axes)), each = n)
      colnames(sc) <- colnames(sol$axes)
      cen <- sweep(means_o, 2L, grand_o, "-") %*% sol$axes

      # 6. store
      used <- unique(assign)
      private$.spec <- Filter(function(s) s$name %in% used, spec)
      private$.grand <- grand_o
      private$.y_train <- y
      self$coefficients <- sol$coef
      self$intercept <- sol$intercept
      self$classes <- levels(y)
      self$priors <- pri
      self$class_counts <- stats::setNames(st$cnt, levels(y))
      self$means <- means_o
      self$features <- colnames(Xm)
      self$variables <- used
      self$dropped <- dropped
      self$wilks <- sol$wilks
      self$wilks_test <- c(wilks = sol$wilks, sol$test)
      self$var_importance <- vi
      self$axes <- sol$axes
      self$eigenvalues <- sol$eigenvalues
      self$canonical_cor <- sol$canonical_cor
      self$scores <- sc
      self$centroids <- cen
      self$confusion <- conf
      self$accuracy <- sum(diag(conf)) / sum(conf)
      self$cv_results <- NULL
      self$selection_history <- sel_hist
      self$n_obs <- n
      self$rows_used <- d$rows
      private$.fitted <- TRUE
      invisible(self)
    },

    #' @description Posterior probabilities of the classes.
    #' @param X A data frame containing at least the variables used by the
    #'   model (column order and extra columns are irrelevant).
    #' @return A numeric matrix with one row per row of `X` and one column per
    #'   class. Rows with missing values or unseen factor levels get `NA`.
    predict_proba = function(X) {
      private$check_fitted()
      dz <- private$design(X)
      K <- length(self$classes)
      P <- matrix(NA_real_, nrow(dz$Xm), K, dimnames = list(NULL, self$classes))
      if (any(dz$ok)) {
        S <- dz$Xm[dz$ok, , drop = FALSE] %*% self$coefficients
        S <- S + rep(self$intercept, each = nrow(S))
        mx <- S[cbind(seq_len(nrow(S)), max.col(S, ties.method = "first"))]
        Pk <- exp(S - mx)                       # stable softmax
        P[dz$ok, ] <- Pk / rowSums(Pk)
      }
      P
    },

    #' @description Predicted class of each observation.
    #' @param X A data frame, as in `$predict_proba()`.
    #' @return A factor with the levels of the training classes (`NA` for rows
    #'   that cannot be classified).
    predict = function(X) {
      P <- self$predict_proba(X)
      out <- rep(NA_character_, nrow(P))
      ok <- rowSums(is.finite(P)) == ncol(P)
      if (any(ok)) {
        out[ok] <- self$classes[max.col(P[ok, , drop = FALSE],
                                         ties.method = "first")]
      }
      factor(out, levels = self$classes)
    },

    #' @description Project observations on the canonical discriminant axes.
    #' @param X A data frame, as in `$predict_proba()`.
    #' @return A numeric matrix with one column per axis (`LD1`, `LD2`, ...).
    transform = function(X) {
      private$check_fitted()
      dz <- private$design(X)
      out <- matrix(NA_real_, nrow(dz$Xm), ncol(self$axes),
                    dimnames = list(NULL, colnames(self$axes)))
      if (any(dz$ok)) {
        Z <- dz$Xm[dz$ok, , drop = FALSE] %*% self$axes
        out[dz$ok, ] <- Z - rep(drop(crossprod(private$.grand, self$axes)), each = nrow(Z))
      }
      out
    },

    #' @description Stratified k-fold cross-validation. A new model with the
    #'   same settings (including variable selection) is fitted on each fold.
    #' @param X,y Data as in `$fit()`.
    #' @param folds Number of folds (at least 2). It is capped at the size of
    #'   the smallest class so every training fold contains all classes.
    #' @param seed Optional integer; sets the global random seed before the
    #'   folds are drawn.
    #' @return Invisibly, a list with `accuracy`, `error_rate`, `confusion`,
    #'   `per_class`, `folds` and `n`; also stored in `$cv_results`.
    cv = function(X, y, folds = 10L, seed = NULL) {
      prm <- private$.params
      d <- private$prepare_xy(X, y)
      X <- d$X
      y <- d$y
      n <- nrow(X)
      folds <- suppressWarnings(as.integer(folds))
      if (length(folds) != 1L || is.na(folds) || folds < 2L) {
        stop("'folds' must be an integer >= 2.", call. = FALSE)
      }
      class_sizes <- table(y)
      max_folds <- min(class_sizes)
      if (max_folds < 2L) {
        stop("Cross-validation requires at least two observations in every class.",
             call. = FALSE)
      }
      if (folds > max_folds) {
        warning("Reducing 'folds' to ", max_folds,
                " so every training fold contains all classes.", call. = FALSE)
        folds <- max_folds
      }
      train_sizes <- class_sizes - ceiling(class_sizes / folds)
      if (sum(train_sizes) <= length(class_sizes)) {
        stop("Each cross-validation training fold must contain more observations ",
             "than the number of classes.", call. = FALSE)
      }
      if (!is.null(seed)) set.seed(seed)
      fold <- integer(n)
      for (lv in levels(y)) {
        id <- which(y == lv)
        if (length(id) > 0L) fold[id] <- sample(rep_len(seq_len(folds), length(id)))
      }
      pred <- factor(rep(NA_character_, n), levels = levels(y))
      for (f in seq_len(folds)) {
        te <- which(fold == f)
        if (length(te) == 0L) next
        tr <- which(fold != f)
        m <- LDAClassifier$new(priors = prm$priors, selection = prm$selection,
                               alpha_enter = prm$alpha_enter, alpha_stay = prm$alpha_stay)
        suppressMessages(suppressWarnings(m$fit(X[tr, , drop = FALSE], y[tr])))
        pred[te] <- m$predict(X[te, , drop = FALSE])
      }
      cm <- table(Observed = y, Predicted = pred)
      res <- list(accuracy = sum(diag(cm)) / sum(cm),
                  error_rate = 1 - sum(diag(cm)) / sum(cm),
                  confusion = cm, per_class = .class_metrics(cm),
                  folds = folds, n = sum(cm))
      self$cv_results <- res
      invisible(res)
    },

    #' @description Per-class recall, precision and F1 score.
    #' @param confusion A confusion matrix (observed in rows, predicted in
    #'   columns). Defaults to the resubstitution matrix `$confusion`.
    #' @return A data frame with one row per class.
    class_metrics = function(confusion = NULL) {
      private$check_fitted()
      .class_metrics(if (is.null(confusion)) self$confusion else confusion)
    },

    #' @description Describe the input variables required by `$predict()`
    #'   (used by the Shiny application to build its input form).
    #' @return A list with, for each variable, `name`, `type` (`"numeric"` or
    #'   `"factor"`), `levels` (factors) or `default` (training mean, numeric).
    describe_inputs = function() {
      private$check_fitted()
      lapply(private$.spec, function(s) {
        if (s$type == "numeric") {
          list(name = s$name, type = "numeric",
               default = unname(private$.grand[s$name]))
        } else {
          list(name = s$name, type = "factor", levels = s$levels)
        }
      })
    },

    #' @description Short description of the model.
    #' @param ... Ignored.
    #' @return The object, invisibly.
    print = function(...) {
      if (!isTRUE(private$.fitted)) {
        cat("<LDAClassifier> not fitted yet. Use $fit(X, y).\n")
        return(invisible(self))
      }
      K <- length(self$classes)
      cat("<LDAClassifier> Linear Discriminant Analysis\n")
      cat(sprintf("  Observations : %d | Classes : %d (%s)\n", self$n_obs, K,
                  paste(self$classes, collapse = ", ")))
      cat(sprintf("  Predictors   : %d variable(s), %d encoded column(s)\n",
                  length(self$variables), length(self$features)))
      cat("  Priors       : ",
          paste(sprintf("%s=%.3f", self$classes, self$priors), collapse = "  "), "\n", sep = "")
      wt <- self$wilks_test
      cat(sprintf("  Wilks' lambda: %.4g (Rao F = %.4g on %g and %.4g df, p %s)\n",
                  wt[["wilks"]], wt[["F"]], wt[["df1"]], wt[["df2"]],
                  format.pval(wt[["p_value"]], digits = 3, eps = 1e-16)))
      cat(sprintf("  Resubstitution accuracy: %.4f", self$accuracy))
      if (!is.null(self$cv_results)) {
        cat(sprintf(" | %d-fold CV accuracy: %.4f", self$cv_results$folds,
                    self$cv_results$accuracy))
      }
      cat("\n")
      if (length(self$dropped) > 0L) {
        cat("  Dropped column(s): ", paste(self$dropped, collapse = ", "), "\n", sep = "")
      }
      invisible(self)
    },

    #' @description Detailed report: priors, classification functions, global
    #'   test, variable importance, discriminant axes, confusion matrix,
    #'   per-class metrics, cross-validation and selection steps.
    #' @param digits Number of significant digits.
    #' @param ... Ignored.
    #' @return Invisibly, a list with the printed components.
    summary = function(digits = 4, ...) {
      private$check_fitted()
      line <- function(txt) cat("\n", txt, "\n", strrep("-", nchar(txt)), "\n", sep = "")
      cat("LINEAR DISCRIMINANT ANALYSIS - SUMMARY\n")
      cat(sprintf("Observations: %d | Classes: %d | Variables: %d (%d encoded columns)\n",
                  self$n_obs, length(self$classes), length(self$variables),
                  length(self$features)))

      line("Classes")
      cls <- data.frame(class = self$classes, count = as.integer(self$class_counts),
                        prior = as.numeric(self$priors), stringsAsFactors = FALSE)
      print(cls, row.names = FALSE, digits = digits)

      line("Global test of class separation (Wilks' lambda, Rao's F)")
      wt <- self$wilks_test
      cat(sprintf("Lambda = %s | F(%s, %s) = %s | p-value %s\n",
                  format(wt[["wilks"]], digits = digits), format(wt[["df1"]]),
                  format(wt[["df2"]], digits = digits), format(wt[["F"]], digits = digits),
                  format.pval(wt[["p_value"]], digits = 3, eps = 1e-16)))

      line("Classification functions (score_k = intercept_k + sum_j coef_jk * x_j)")
      print(round(rbind(self$coefficients, "(Constant)" = self$intercept), digits))

      line("Variable importance (partial Wilks' lambda given the other variables)")
      vi <- self$var_importance
      vi_show <- vi
      vi_show$p_value <- format.pval(vi$p_value, digits = 3, eps = 1e-16)
      print(vi_show, row.names = FALSE, digits = digits)

      line("Canonical discriminant axes")
      ax <- data.frame(axis = colnames(self$axes), eigenvalue = self$eigenvalues,
                       pct_variance = 100 * self$eigenvalues / sum(self$eigenvalues),
                       cum_pct = cumsum(100 * self$eigenvalues / sum(self$eigenvalues)),
                       canonical_cor = self$canonical_cor, stringsAsFactors = FALSE)
      print(ax, row.names = FALSE, digits = digits)

      line("Resubstitution confusion matrix (observed in rows)")
      print(self$confusion)
      cat(sprintf("Accuracy: %.4f | Error rate: %.4f\n", self$accuracy, 1 - self$accuracy))
      mt <- .class_metrics(self$confusion)
      print(mt, row.names = FALSE, digits = digits)

      if (!is.null(self$cv_results)) {
        cv <- self$cv_results
        line(sprintf("%d-fold cross-validation", cv$folds))
        print(cv$confusion)
        cat(sprintf("CV accuracy: %.4f | CV error rate: %.4f\n", cv$accuracy, cv$error_rate))
      }
      if (!is.null(self$selection_history)) {
        line(sprintf("Variable selection (%s)", private$.params$selection))
        print(self$selection_history, row.names = FALSE, digits = digits)
      }
      if (length(self$dropped) > 0L) {
        cat("\nDropped (constant or collinear) column(s): ",
            paste(self$dropped, collapse = ", "), "\n", sep = "")
      }
      invisible(list(classes = cls, wilks_test = self$wilks_test,
                     coefficients = rbind(self$coefficients, "(Constant)" = self$intercept),
                     var_importance = vi, axes = ax, confusion = self$confusion,
                     class_metrics = mt, cv = self$cv_results,
                     selection = self$selection_history))
    },

    #' @description Base-graphics plots.
    #' @param type `"axes"` (observations and class centroids on two
    #'   discriminant axes; box plot if there is only one axis),
    #'   `"importance"` (partial F of each variable) or `"confusion"`
    #'   (mosaic plot of the resubstitution confusion matrix).
    #' @param dims Two distinct valid axes to display for `type = "axes"`.
    #' @param max_points Maximum number of observations drawn (random subset);
    #'   must be positive.
    #' @return The object, invisibly.
    plot = function(type = c("axes", "importance", "confusion"), dims = c(1L, 2L),
                    max_points = 5000L) {
      private$check_fitted()
      type <- match.arg(type)
      if (length(max_points) != 1L || !is.finite(max_points) ||
          max_points < 1) {
        stop("'max_points' must be a positive number.", call. = FALSE)
      }
      max_points <- as.integer(max_points)
      K <- length(self$classes)
      cols <- grDevices::hcl.colors(max(K, 2L), palette = "Dark 3")[seq_len(K)]
      if (type == "axes") {
        r <- ncol(self$scores)
        y <- private$.y_train
        n <- nrow(self$scores)
        idx <- if (n > max_points) sort(sample.int(n, max_points)) else seq_len(n)
        pct <- 100 * self$eigenvalues / sum(self$eigenvalues)
        if (r >= 2L) {
          if (length(dims) != 2L || anyNA(dims) ||
              any(dims < 1L | dims > r) || anyDuplicated(dims)) {
            stop("'dims' must contain two distinct valid axis numbers.",
                 call. = FALSE)
          }
          d <- as.integer(dims)
          graphics::plot(self$scores[idx, d[1L]], self$scores[idx, d[2L]],
                         col = grDevices::adjustcolor(cols[as.integer(y[idx])], 0.6),
                         pch = 19, cex = 0.8,
                         xlab = sprintf("LD%d (%.1f%%)", d[1L], pct[d[1L]]),
                         ylab = sprintf("LD%d (%.1f%%)", d[2L], pct[d[2L]]),
                         main = "Discriminant axes")
          graphics::abline(h = 0, v = 0, lty = 3, col = "grey60")
          graphics::points(self$centroids[, d[1L]], self$centroids[, d[2L]],
                           pch = 23, bg = cols, col = "black", cex = 2)
          graphics::text(self$centroids[, d[1L]], self$centroids[, d[2L]],
                         labels = self$classes, pos = 3, font = 2)
          graphics::legend("topright", legend = self$classes, col = cols, pch = 19, bty = "n")
        } else {
          graphics::boxplot(self$scores[idx, 1L] ~ y[idx], col = cols, xlab = "Class",
                            ylab = sprintf("LD1 (%.1f%%)", pct[1L]),
                            main = "Discriminant axis")
        }
      } else if (type == "importance") {
        vi <- self$var_importance
        vi <- vi[order(vi$F), , drop = FALSE]
        op <- graphics::par(mar = c(4.5, max(6, 0.6 * max(nchar(vi$variable))), 3, 1))
        on.exit(graphics::par(op), add = TRUE)
        graphics::barplot(vi$F, names.arg = vi$variable, horiz = TRUE, las = 1,
                          col = "steelblue",
                          xlab = "Partial F (contribution given the other variables)",
                          main = "Variable importance")
      } else {
        graphics::mosaicplot(self$confusion, color = cols, las = 1,
                             main = "Confusion matrix (resubstitution)")
      }
      invisible(self)
    }
  ),

  private = list(
    .params = NULL,
    .spec = NULL,
    .grand = NULL,
    .y_train = NULL,
    .fitted = FALSE,

    check_fitted = function() {
      if (!isTRUE(private$.fitted)) {
        stop("The model has not been fitted yet: call $fit(X, y) first.", call. = FALSE)
      }
      invisible(TRUE)
    },

    # validate X and y, remove incomplete rows
    prepare_xy = function(X, y) {
      if (is.matrix(X)) X <- as.data.frame(X)
      if (!is.data.frame(X)) stop("'X' must be a data.frame.", call. = FALSE)
      X <- as.data.frame(X)
      if (ncol(X) < 1L) stop("'X' must have at least one column.", call. = FALSE)
      if (anyDuplicated(names(X)) > 0L || any(!nzchar(names(X)))) {
        stop("Column names of 'X' must be unique and non-empty.", call. = FALSE)
      }
      if (is.character(y) || is.logical(y)) y <- factor(y)
      if (!is.factor(y)) stop("'y' must be a factor.", call. = FALSE)
      if (length(y) != nrow(X)) {
        stop("'X' and 'y' must have the same number of observations.", call. = FALSE)
      }
      ok <- stats::complete.cases(X) & !is.na(y)
      if (!all(ok)) {
        if (private$.params$na_action == "fail") {
          stop("Missing values found in 'X' or 'y' (na_action = \"fail\").", call. = FALSE)
        }
        message(sum(!ok), " observation(s) with missing values removed.")
      }
      rows <- which(ok)
      X_ok <- X[rows, , drop = FALSE]
      y_ok <- droplevels(y[rows])
      if (nrow(X_ok) == 0L) {
        stop("No complete observation remains after removing missing values.",
             call. = FALSE)
      }
      list(X = X_ok, y = y_ok, rows = rows)
    },

    resolve_priors = function(cnt, classes) {
      p <- private$.params$priors
      K <- length(classes)
      if (is.null(p)) {
        pr <- cnt / sum(cnt)
      } else if (is.character(p)) {
        pr <- rep(1 / K, K)
      } else {
        if (length(p) != K) {
          stop("'priors' must have one value per class (", K, " expected).", call. = FALSE)
        }
        if (!is.null(names(p))) {
          if (!all(classes %in% names(p))) {
            stop("Names of 'priors' must match the classes: ",
                 paste(classes, collapse = ", "), call. = FALSE)
          }
          p <- p[classes]
        }
        pr <- as.numeric(p) / sum(p)
      }
      names(pr) <- classes
      pr
    },

    # encode new data exactly as the training data
    design = function(X) {
      if (is.matrix(X)) X <- as.data.frame(X)
      if (!is.data.frame(X)) stop("'X' must be a data.frame.", call. = FALSE)
      X <- as.data.frame(X)
      .check_columns(X, private$.spec)
      enc <- .encode_matrix(X, private$.spec)
      if (any(enc$bad_rows)) {
        warning(sum(enc$bad_rows), " observation(s) with factor levels not seen ",
                "during training: output set to NA.", call. = FALSE)
      }
      Xm <- enc$Xm[, self$features, drop = FALSE]
      list(Xm = Xm, ok = is.finite(rowSums(Xm)))
    }
  )
)
