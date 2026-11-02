# ---------------------------------------------------------------------------
# Shiny application of the 'ldapkg' package: predictive linear discriminant
# analysis. Launch with ldapkg::run_app().
# Required: shiny, DT. Optional: readxl (xlsx files), data.table (fast fread).
# ---------------------------------------------------------------------------
library(shiny)
library(ldapkg)

options(shiny.maxRequestSize = 500 * 1024^2)       # files up to 500 MB

# ----------------------------------------------------------------- helpers
count_chr <- function(ch, s) nchar(s) - nchar(gsub(ch, "", s, fixed = TRUE))

# guess the separator from the header line (tab preferred on ties)
sniff_sep <- function(path) {
  l <- tryCatch(readLines(path, n = 1L, warn = FALSE), error = function(e) "")
  if (length(l) == 0L || is.na(l[1L])) return("\t")
  seps <- c("\t", ";", ",")
  cnt <- vapply(seps, count_chr, numeric(1L), s = l[1L])
  if (max(cnt) == 0) "\t" else seps[which.max(cnt)]
}

stratified_split <- function(y, test_pct, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  test <- integer(0L)
  for (lv in levels(factor(y))) {
    id <- which(y == lv)
    if (length(id) > 1L && test_pct > 0) {
      n_test <- min(length(id) - 1L, round(length(id) * test_pct / 100))
      if (n_test > 0L) test <- c(test, sample(id, n_test))
    }
  }
  test <- sort(test)
  list(test = test, train = setdiff(seq_along(y), test))
}

# read csv / tsv / txt / xlsx into a plain data.frame
read_data <- function(path, name, sep = "auto", dec = ".") {
  ext <- tolower(tools::file_ext(name))
  if (ext %in% c("xlsx", "xls")) {
    if (!requireNamespace("readxl", quietly = TRUE)) {
      stop("Package 'readxl' is required to read Excel files.")
    }
    return(as.data.frame(readxl::read_excel(path)))
  }
  if (identical(sep, "auto")) sep <- sniff_sep(path)
  d <- if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::fread(path, sep = sep, dec = dec, data.table = FALSE,
                      na.strings = c("NA", ""), stringsAsFactors = FALSE)
  } else {
    utils::read.table(path, header = TRUE, sep = sep, dec = dec, quote = "\"",
                      comment.char = "", na.strings = c("NA", ""),
                      stringsAsFactors = FALSE, check.names = FALSE, fill = TRUE)
  }
  as.data.frame(d)
}

is_cat <- function(v) is.character(v) || is.factor(v) || is.logical(v)

css <- "
.kpi-row{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 14px}
.kpi{background:#f4f7fb;border-left:4px solid #2c7be5;border-radius:4px;padding:8px 14px;min-width:120px}
.kpi-v{font-size:22px;font-weight:600;line-height:1.1}
.kpi-l{font-size:12px;color:#5b6b7b}
.big-class{font-size:30px;font-weight:700;color:#2c7be5;margin:8px 0}
"

# ---------------------------------------------------------------------- UI
ui <- fluidPage(
  tags$head(tags$style(HTML(css))),
  titlePanel("ldapkg - Predictive Linear Discriminant Analysis"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("1. Data"),
      radioButtons("src", NULL, c("Example dataset" = "example", "Upload a file" = "file"),
                   inline = TRUE),
      conditionalPanel("input.src == 'file'",
        fileInput("file", NULL, accept = c(".csv", ".tsv", ".txt", ".xlsx", ".xls"),
                  placeholder = "csv / tsv / txt / xlsx"),
        fluidRow(
          column(6, selectInput("sep", "Separator",
                                c("Auto" = "auto", "Tab" = "\t", "Semicolon" = ";", "Comma" = ","))),
          column(6, selectInput("dec", "Decimal", c(".", ","))))),
      conditionalPanel("input.src == 'example'",
        selectInput("example", NULL, choices = ldapkg_example(), selected = "wine")),
      h4("2. Variables"),
      uiOutput("var_ui"),
      h4("3. Options"),
      radioButtons("priors", "Prior probabilities",
                   c("Class frequencies" = "emp", "Equal" = "equal"), inline = TRUE),
      selectInput("selection", "Variable selection",
                  c("None" = "none", "Forward" = "forward", "Backward" = "backward",
                    "Stepwise" = "stepwise")),
      conditionalPanel("input.selection == 'forward' || input.selection == 'stepwise'",
        numericInput("alpha_enter", "alpha to enter", 0.05, min = 0.0001, max = 1, step = 0.01)),
      conditionalPanel("input.selection == 'backward' || input.selection == 'stepwise'",
        numericInput("alpha_stay", "alpha to stay", 0.05, min = 0.0001, max = 1, step = 0.01)),
      sliderInput("test_pct", "Hold-out test sample (%)", 0, 50, 0, step = 5),
      numericInput("cv_folds", "Cross-validation folds (0 = none)", 0, min = 0, max = 20),
      numericInput("seed", "Random seed", 42),
      actionButton("run", "Fit the model", class = "btn-primary btn-block", icon = icon("play"))
    ),
    mainPanel(
      width = 9,
      tabsetPanel(
        id = "tabs",
        tabPanel("Data", br(), verbatimTextOutput("data_info"), DT::DTOutput("data_tbl")),
        tabPanel("Model", br(), uiOutput("kpis"), verbatimTextOutput("summary_txt")),
        tabPanel("Coefficients", br(),
                 p("Classification functions: the observation is assigned to the class with the highest score."),
                 DT::DTOutput("coef_tbl"),
                 conditionalPanel("output.has_selection", h4("Variable selection steps"),
                                  DT::DTOutput("sel_tbl"))),
        tabPanel("Variables", br(),
                 fluidRow(column(6, plotOutput("imp_plot", height = "420px")),
                          column(6, DT::DTOutput("imp_tbl")))),
        tabPanel("Discriminant axes", br(), uiOutput("dims_ui"),
                 plotOutput("axes_plot", height = "480px"), DT::DTOutput("eig_tbl")),
        tabPanel("Performance", br(),
                 fluidRow(column(6, h4("Training sample"), plotOutput("cm_plot", height = "320px"),
                                 DT::DTOutput("cm_tbl")),
                          column(6, h4("Per-class metrics"), DT::DTOutput("met_tbl"))),
                 uiOutput("test_ui"), uiOutput("cv_ui")),
        tabPanel("Illustrative variables", br(),
                 p("Variables not used to fit the model, related to the discriminant axes (training rows)."),
                 uiOutput("ill_msg"),
                 fluidRow(column(6, plotOutput("ill_plot", height = "420px")),
                          column(6, h4("Numeric: correlation with the axes"), DT::DTOutput("ill_num"),
                                 h4("Categorical: modality centroids"), DT::DTOutput("ill_cat")))),
        tabPanel("Deploy",
                 br(),
                 tabsetPanel(
                   tabPanel("One individual", br(),
                            fluidRow(column(4, uiOutput("indiv_ui")),
                                     column(8, uiOutput("one_class"), plotOutput("one_plot", height = "300px"),
                                            DT::DTOutput("one_tbl")))),
                   tabPanel("From a file", br(),
                            fluidRow(
                              column(4,
                                fileInput("newfile", "New individuals (same variables)",
                                          accept = c(".csv", ".tsv", ".txt", ".xlsx", ".xls")),
                                fluidRow(
                                  column(6, selectInput("sep_new", "Separator",
                                                        c("Auto" = "auto", "Tab" = "\t", "Semicolon" = ";", "Comma" = ","))),
                                  column(6, selectInput("dec_new", "Decimal", c(".", ",")))),
                                downloadButton("dl", "Download predictions (csv)")),
                              column(8, verbatimTextOutput("batch_info"), DT::DTOutput("batch_tbl"))))))
      )
    )
  )
)

# ------------------------------------------------------------------ server
server <- function(input, output, session) {

  # ---- data
  data_raw <- reactive({
    if (identical(input$src, "example")) {
      req(input$example)
      ldapkg_example(input$example)
    } else {
      req(input$file)
      tryCatch(read_data(input$file$datapath, input$file$name, input$sep, input$dec),
               error = function(e) {
                 showNotification(paste("Cannot read the file:", conditionMessage(e)),
                                  type = "error", duration = 10)
                 NULL
               })
    }
  })

  output$data_info <- renderText({
    d <- req(data_raw())
    paste0(nrow(d), " rows x ", ncol(d), " columns | ",
           sum(vapply(d, is.numeric, logical(1L))), " numeric, ",
           sum(vapply(d, is_cat, logical(1L))), " categorical | ",
           sum(!stats::complete.cases(d)), " row(s) with missing values")
  })
  output$data_tbl <- DT::renderDT(
    DT::datatable(req(data_raw()), options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
  )

  # ---- variable choice (target / explanatory / illustrative)
  output$var_ui <- renderUI({
    d <- req(data_raw())
    nm <- names(d)
    cat_like <- nm[vapply(d, function(v) is_cat(v) || length(unique(v)) <= 10L, logical(1L))]
    def <- if (length(cat_like)) utils::tail(cat_like, 1L) else utils::tail(nm, 1L)
    others <- setdiff(nm, def)
    tagList(
      selectInput("target", "Target (classes)", choices = nm, selected = def),
      selectizeInput("xvars", "Explanatory variables", choices = others, selected = others,
                     multiple = TRUE),
      selectizeInput("ivars", "Illustrative variables", choices = others, selected = NULL,
                     multiple = TRUE)
    )
  })
  observeEvent(input$target, {
    d <- data_raw()
    req(d, input$target %in% names(d))
    nm <- setdiff(names(d), input$target)
    iv <- intersect(input$ivars, nm)
    updateSelectizeInput(session, "xvars", choices = nm, selected = setdiff(nm, iv))
    updateSelectizeInput(session, "ivars", choices = nm, selected = iv)
  }, ignoreInit = TRUE)

  # ---- fitting
  res <- reactiveVal(NULL)
  observeEvent(input$run, {
    d <- data_raw()
    if (is.null(d) || is.null(input$target) || length(input$xvars) == 0L) {
      showNotification("Choose a target and at least one explanatory variable.", type = "error")
      return()
    }
    ivars <- setdiff(input$ivars, input$xvars)
    seed <- if (is.na(input$seed)) 1 else input$seed
    out <- NULL
    withProgress(message = "Fitting the LDA model...", value = 0.5, {
      out <- tryCatch(
        withCallingHandlers({
          y <- factor(d[[input$target]])
          X <- d[, input$xvars, drop = FALSE]
          n <- nrow(d)
          split <- stratified_split(y, input$test_pct, seed)
          test_idx <- split$test
          train_idx <- split$train
          m <- LDAClassifier$new(priors = if (identical(input$priors, "equal")) "equal" else NULL,
                                 selection = input$selection,
                                 alpha_enter = if (is.na(input$alpha_enter)) 0.05 else input$alpha_enter,
                                 alpha_stay = if (is.na(input$alpha_stay)) 0.05 else input$alpha_stay)
          m$fit(X[train_idx, , drop = FALSE], y[train_idx])
          r <- list(model = m, target = input$target, ivars = ivars,
                    train_rows = train_idx[m$rows_used], test = NULL)
          if (length(test_idx) > 0L) {
            pr <- m$predict(X[test_idx, , drop = FALSE])
            truth <- factor(as.character(y[test_idx]), levels = m$classes)
            cm <- table(Observed = truth, Predicted = pr)
            n_classified <- sum(cm)
            r$test <- list(
              confusion = cm,
              accuracy = if (n_classified > 0L) sum(diag(cm)) / n_classified else NA_real_,
              n = length(test_idx),
              classified = n_classified,
              coverage = if (length(test_idx) > 0L) n_classified / length(test_idx) else NA_real_
            )
          }
          if (!is.na(input$cv_folds) && input$cv_folds >= 2L) {
            setProgress(0.8, message = "Cross-validation...")
            m$cv(X[train_idx, , drop = FALSE], y[train_idx], folds = input$cv_folds, seed = seed)
          }
          r
        },
        warning = function(w) {
          showNotification(conditionMessage(w), type = "warning", duration = 8)
          invokeRestart("muffleWarning")
        },
        message = function(mc) {
          showNotification(trimws(conditionMessage(mc)), type = "message", duration = 6)
          invokeRestart("muffleMessage")
        }),
        error = function(e) {
          showNotification(conditionMessage(e), type = "error", duration = 12)
          NULL
        })
    })
    res(out)
    if (!is.null(out)) updateTabsetPanel(session, "tabs", selected = "Model")
  })

  model <- reactive(req(res())$model)

  # ---- model overview
  output$kpis <- renderUI({
    r <- req(res())
    m <- r$model
    kp <- function(label, value) div(class = "kpi", div(class = "kpi-v", value), div(class = "kpi-l", label))
    div(class = "kpi-row",
        kp("Observations", m$n_obs), kp("Classes", length(m$classes)),
        kp("Variables", length(m$variables)),
        kp("Wilks' lambda", formatC(m$wilks, digits = 4, format = "g")),
        kp("Global F p-value", format.pval(m$wilks_test[["p_value"]], digits = 3, eps = 1e-16)),
        kp("Train accuracy", sprintf("%.1f%%", 100 * m$accuracy)),
        if (!is.null(r$test)) kp("Test accuracy", sprintf("%.1f%%", 100 * r$test$accuracy)),
        if (!is.null(m$cv_results)) kp("CV accuracy", sprintf("%.1f%%", 100 * m$cv_results$accuracy)))
  })
  output$summary_txt <- renderPrint({ model()$summary() })

  # ---- coefficients and selection
  output$coef_tbl <- DT::renderDT({
    m <- model()
    cf <- rbind(m$coefficients, "(Constant)" = m$intercept)
    DT::datatable(round(cf, 5), options = list(pageLength = 15, scrollX = TRUE))
  })
  output$has_selection <- reactive({ !is.null(res()) && !is.null(res()$model$selection_history) })
  outputOptions(output, "has_selection", suspendWhenHidden = FALSE)
  output$sel_tbl <- DT::renderDT({
    h <- req(model()$selection_history)
    DT::formatSignif(DT::datatable(h, rownames = FALSE, options = list(dom = "t")), c("F", "p_value", "wilks"), 4)
  })

  # ---- variables
  output$imp_plot <- renderPlot(model()$plot("importance"))
  output$imp_tbl <- DT::renderDT({
    DT::formatSignif(DT::datatable(model()$var_importance, rownames = FALSE, options = list(pageLength = 15, scrollX = TRUE)), c("partial_wilks", "F", "df2", "p_value", "tolerance"), 4)
  })

  # ---- axes
  output$dims_ui <- renderUI({
    r <- ncol(model()$axes)
    if (r < 2L) return(NULL)
    fluidRow(column(3, selectInput("dim_x", "x axis", 1:r, selected = 1)),
             column(3, selectInput("dim_y", "y axis", 1:r, selected = 2)))
  })
  output$axes_plot <- renderPlot({
    m <- model()
    dims <- if (ncol(m$axes) >= 2L && !is.null(input$dim_x)) as.integer(c(input$dim_x, input$dim_y)) else c(1L, 2L)
    m$plot("axes", dims = dims)
  })
  output$eig_tbl <- DT::renderDT({
    m <- model()
    ax <- data.frame(axis = colnames(m$axes), eigenvalue = m$eigenvalues,
                     pct_variance = 100 * m$eigenvalues / sum(m$eigenvalues),
                     canonical_cor = m$canonical_cor)
    DT::formatSignif(DT::datatable(ax, rownames = FALSE, options = list(dom = "t")), c("eigenvalue", "pct_variance", "canonical_cor"), 4)
  })

  # ---- performance
  output$cm_plot <- renderPlot(model()$plot("confusion"))
  output$cm_tbl <- DT::renderDT({
    DT::datatable(as.data.frame.matrix(unclass(model()$confusion)), options = list(dom = "t"))
  })
  output$met_tbl <- DT::renderDT({
    DT::formatSignif(DT::datatable(model()$class_metrics(), rownames = FALSE, options = list(dom = "t")), c("recall", "precision", "f1"), 3)
  })
  output$test_ui <- renderUI({
    r <- req(res())
    if (is.null(r$test)) return(NULL)
    tagList(hr(), h4(sprintf("Hold-out test sample (n = %d): accuracy %.2f%%, coverage %.2f%%",
                             r$test$n, 100 * r$test$accuracy, 100 * r$test$coverage)),
            fluidRow(column(6, DT::DTOutput("test_cm")), column(6, DT::DTOutput("test_met"))))
  })
  output$test_cm <- DT::renderDT({
    DT::datatable(as.data.frame.matrix(unclass(req(res()$test)$confusion)), options = list(dom = "t"))
  })
  output$test_met <- DT::renderDT({
    DT::formatSignif(DT::datatable(model()$class_metrics(req(res()$test)$confusion), rownames = FALSE, options = list(dom = "t")), c("recall", "precision", "f1"), 3)
  })
  output$cv_ui <- renderUI({
    cv <- req(res())$model$cv_results
    if (is.null(cv)) return(NULL)
    tagList(hr(), h4(sprintf("%d-fold cross-validation: accuracy %.2f%%", cv$folds, 100 * cv$accuracy)),
            fluidRow(column(6, DT::DTOutput("cv_cm")), column(6, DT::DTOutput("cv_met"))))
  })
  output$cv_cm <- DT::renderDT({
    DT::datatable(as.data.frame.matrix(unclass(req(model()$cv_results)$confusion)), options = list(dom = "t"))
  })
  output$cv_met <- DT::renderDT({
    DT::formatSignif(DT::datatable(req(model()$cv_results)$per_class, rownames = FALSE, options = list(dom = "t")), c("recall", "precision", "f1"), 3)
  })

  # ---- illustrative variables (rows aligned with the training scores)
  ill_data <- reactive({
    r <- req(res())
    if (length(r$ivars) == 0L) return(NULL)
    data_raw()[r$train_rows, r$ivars, drop = FALSE]
  })
  output$ill_msg <- renderUI({
    if (is.null(ill_data())) div(class = "alert alert-info",
                                 "No illustrative variable selected: choose some in the sidebar, then fit again.")
  })
  output$ill_num <- DT::renderDT({
    d <- req(ill_data()); sc <- model()$scores
    num <- names(d)[vapply(d, is.numeric, logical(1L))]
    req(length(num) > 0L)
    tab <- do.call(rbind, lapply(num, function(v) {
      ok <- !is.na(d[[v]])
      if (sum(ok) < 3L || stats::sd(d[[v]][ok]) == 0) return(rep(NA_real_, ncol(sc)))
      as.numeric(stats::cor(d[[v]][ok], sc[ok, , drop = FALSE]))
    }))
    dimnames(tab) <- list(num, colnames(sc))
    DT::datatable(round(tab, 3), options = list(dom = "t"))
  })
  output$ill_cat <- DT::renderDT({
    d <- req(ill_data()); sc <- model()$scores
    cats <- names(d)[vapply(d, is_cat, logical(1L))]
    req(length(cats) > 0L)
    rows <- lapply(cats, function(v) {
      f <- as.character(d[[v]])
      ok <- !is.na(f)
      sums <- rowsum(sc[ok, , drop = FALSE], f[ok])
      cnt <- as.vector(table(factor(f[ok], levels = rownames(sums))))
      cen <- sums / cnt
      data.frame(variable = v, modality = rownames(cen), n = cnt, round(cen, 3),
                 row.names = NULL, check.names = FALSE)
    })
    DT::datatable(do.call(rbind, rows), rownames = FALSE, options = list(pageLength = 10, scrollX = TRUE))
  })
  output$ill_plot <- renderPlot({
    d <- req(ill_data()); sc <- model()$scores
    req(ncol(sc) >= 2L)
    num <- names(d)[vapply(d, is.numeric, logical(1L))]
    req(length(num) > 0L)
    co <- do.call(rbind, lapply(num, function(v) {
      ok <- !is.na(d[[v]])
      if (sum(ok) < 3L || stats::sd(d[[v]][ok]) == 0) return(c(0, 0))
      as.numeric(stats::cor(d[[v]][ok], sc[ok, 1:2, drop = FALSE]))
    }))
    th <- seq(0, 2 * pi, length.out = 200)
    plot(cos(th), sin(th), type = "l", asp = 1, xlab = "LD1", ylab = "LD2",
         main = "Correlation circle (illustrative numeric variables)", col = "grey60")
    abline(h = 0, v = 0, lty = 3, col = "grey70")
    arrows(0, 0, co[, 1], co[, 2], length = 0.1, col = "firebrick")
    text(co[, 1], co[, 2], labels = num, pos = 3, col = "firebrick", cex = 0.9)
  })

  # ---- deployment: one individual
  output$indiv_ui <- renderUI({
    info <- model()$describe_inputs()
    ctrls <- lapply(seq_along(info), function(i) {
      s <- info[[i]]
      id <- paste0("ind_", i)
      if (s$type == "numeric") numericInput(id, s$name, value = signif(s$default, 4))
      else selectInput(id, s$name, choices = s$levels, selected = s$levels[1L])
    })
    tagList(h4("Values of the new individual"), ctrls,
            actionButton("predict_one", "Classify", class = "btn-primary"))
  })
  one_pred <- eventReactive(input$predict_one, {
    m <- model()
    info <- m$describe_inputs()
    vals <- lapply(seq_along(info), function(i) {
      v <- input[[paste0("ind_", i)]]
      if (info[[i]]$type == "numeric") as.numeric(v) else as.character(v)
    })
    names(vals) <- vapply(info, function(s) s$name, character(1L))
    nd <- as.data.frame(vals, stringsAsFactors = FALSE, check.names = FALSE)
    P <- m$predict_proba(nd)
    validate(need(!anyNA(P), "Cannot classify: a value is missing."))
    list(class = as.character(m$predict(nd)), prob = P[1L, ])
  })
  output$one_class <- renderUI({
    p <- one_pred()
    tagList(h4("Predicted class"), div(class = "big-class", p$class))
  })
  output$one_plot <- renderPlot({
    p <- one_pred()
    op <- par(mar = c(4, 8, 2, 1)); on.exit(par(op))
    barplot(rev(p$prob), horiz = TRUE, las = 1, xlim = c(0, 1), col = "steelblue",
            xlab = "Posterior probability", main = "Class membership probabilities")
  })
  output$one_tbl <- DT::renderDT({
    p <- one_pred()
    DT::datatable(data.frame(class = names(p$prob), probability = round(as.numeric(p$prob), 4)),
                  rownames = FALSE, options = list(dom = "t"))
  })

  # ---- deployment: file of new individuals
  batch_pred <- reactive({
    m <- model()
    req(input$newfile)
    out <- tryCatch(
      withCallingHandlers({
        nd <- read_data(input$newfile$datapath, input$newfile$name, input$sep_new, input$dec_new)
        P <- m$predict_proba(nd)
        colnames(P) <- paste0("P(", m$classes, ")")
        list(newdata = nd,
             result = cbind(data.frame(predicted_class = m$predict(nd)),
                            as.data.frame(round(P, 5)), nd))
      }, warning = function(w) {
        showNotification(conditionMessage(w), type = "warning", duration = 8)
        invokeRestart("muffleWarning")
      }),
      error = function(e) {
        showNotification(conditionMessage(e), type = "error", duration = 10)
        NULL
      })
    req(out)
    out
  })
  output$batch_info <- renderText({
    b <- batch_pred()
    m <- model()
    txt <- sprintf("%d individual(s) classified; %d could not be classified (missing values / unseen levels).",
                   nrow(b$result), sum(is.na(b$result$predicted_class)))
    tg <- res()$target
    if (tg %in% names(b$newdata)) {
      truth <- as.character(b$newdata[[tg]])
      ok <- !is.na(truth) & !is.na(b$result$predicted_class)
      if (any(ok)) txt <- paste0(txt, sprintf("\nThe file contains '%s': accuracy = %.2f%%.", tg,
                                              100 * mean(truth[ok] == as.character(b$result$predicted_class)[ok])))
    }
    txt
  })
  output$batch_tbl <- DT::renderDT({
    DT::datatable(batch_pred()$result, rownames = FALSE, options = list(pageLength = 10, scrollX = TRUE))
  })
  output$dl <- downloadHandler(
    filename = function() "predictions.csv",
    content = function(file) utils::write.csv(batch_pred()$result, file, row.names = FALSE)
  )
}

shinyApp(ui, server)
