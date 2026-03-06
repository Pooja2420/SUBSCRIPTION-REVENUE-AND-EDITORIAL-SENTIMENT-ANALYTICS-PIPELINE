################################################################################
# NYT COMPANY — SUBSCRIPTION REVENUE & EDITORIAL SENTIMENT ANALYTICS PIPELINE
# Production-Grade R Code
#
# Problem Statement:
#   The New York Times Company has transformed from a print-first publisher into
#   a digital subscription powerhouse. This pipeline models the key drivers of
#   digital subscription revenue using quarterly financials, simulates article-
#   level sentiment scoring, builds a forecasting model, and delivers an
#   interactive Shiny dashboard for executive reporting.
#
# Modules:
#   1. Data Ingestion & Simulation  — realistic NYT financial + editorial data
#   2. Feature Engineering          — sentiment, seasonality, growth metrics
#   3. Exploratory Data Analysis    — ggplot2 publication-quality visuals
#   4. Predictive Modeling          — XGBoost + ARIMA ensemble for revenue forecast
#   5. Model Evaluation             — RMSE, MAE, MAPE, residual diagnostics
#   6. Shiny Dashboard              — interactive executive reporting UI
#
# Author:  Production R Pipeline
# R Version: >= 4.2.0
################################################################################


# ─── 0. ENVIRONMENT SETUP ──────────────────────────────────────────────────────

suppressPackageStartupMessages({
  # Core data wrangling
  library(tidyverse)
  library(lubridate)
  library(scales)

  # Modeling
  library(forecast)    # ARIMA
  library(xgboost)     # Gradient boosting
  library(caret)       # Cross-validation framework
  library(Metrics)     # RMSE, MAE, MAPE

  # Visualization
  library(ggplot2)
  library(patchwork)   # Compose multi-panel plots
  library(ggthemes)
  library(viridis)

  # Shiny dashboard
  library(shiny)
  library(shinydashboard)
  library(plotly)
  library(DT)

  # Utilities
  library(glue)
  library(janitor)
  library(zoo)
})

set.seed(42)

# ─── GLOBAL CONSTANTS ──────────────────────────────────────────────────────────

NYT_BLUE    <- "#326891"
NYT_BLACK   <- "#121212"
NYT_GRAY    <- "#f7f7f5"
NYT_ACCENT  <- "#d0021b"

# ─── LOGGING UTILITY ────────────────────────────────────────────────────────────

log_info <- function(msg, ...) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(glue("[{ts}] INFO  — {msg}\n", ...))
}

log_warn <- function(msg, ...) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(glue("[{ts}] WARN  — {msg}\n", ...))
}

################################################################################
# MODULE 1 — DATA INGESTION & SIMULATION
################################################################################

log_info("MODULE 1: Generating NYT financial & editorial datasets")

# ── 1A. Quarterly Financial Data (Q1 2014 – Q4 2023, 40 quarters) ─────────────
#
# Based on publicly reported NYT financials:
#   • Digital subscription revenue grew from ~$150M (2014) to ~$975M (2023)
#   • Print advertising declined ~60% over the decade
#   • Total digital subscribers: 300K (2014) → 9.7M (2023)
#

generate_financial_data <- function() {

  quarters <- seq.Date(as.Date("2014-01-01"), as.Date("2023-10-01"), by = "quarter")
  n        <- length(quarters)

  # Base trend index (0 → 1)
  t <- seq(0, 1, length.out = n)

  # ── Digital subscription revenue: logistic growth + noise
  dig_sub_rev <- 150 * exp(2.0 * t) +
    rnorm(n, mean = 0, sd = 8) +
    15 * sin(2 * pi * seq_along(t) / 4)   # mild seasonality
  dig_sub_rev <- pmax(dig_sub_rev, 100)

  # ── Digital subscriber count (thousands): S-curve
  dig_subs_k <- 300 + (9700 - 300) / (1 + exp(-8 * (t - 0.5))) +
    rnorm(n, 0, 120)
  dig_subs_k <- pmax(dig_subs_k, 250)

  # ── Print advertising: declining trend
  print_adv <- 350 * exp(-1.8 * t) + rnorm(n, 0, 12)
  print_adv  <- pmax(print_adv, 40)

  # ── Digital advertising: modest growth
  dig_adv <- 80 + 120 * t + rnorm(n, 0, 9)

  # ── Other revenue (Wirecutter, Cooking, Games)
  other_rev <- 20 + 180 * t^2 + rnorm(n, 0, 5)

  # ── Total revenue
  total_rev <- dig_sub_rev + print_adv + dig_adv + other_rev

  # ── Operating expenses
  op_exp <- 0.72 * total_rev + rnorm(n, 0, 15)

  # ── ARPU: Average Revenue Per User (digital, quarterly, $)
  arpu <- (dig_sub_rev * 1e6) / (dig_subs_k * 1000) / 3  # monthly ARPU
  arpu <- round(arpu, 2)

  tibble(
    date               = quarters,
    year               = year(quarters),
    quarter            = quarter(quarters),
    period             = glue("Q{quarter(quarters)} {year(quarters)}"),
    dig_sub_rev_m      = round(dig_sub_rev, 2),   # $M
    dig_subs_k         = round(dig_subs_k),        # thousands
    print_adv_m        = round(print_adv, 2),
    dig_adv_m          = round(dig_adv, 2),
    other_rev_m        = round(other_rev, 2),
    total_rev_m        = round(total_rev, 2),
    op_exp_m           = round(op_exp, 2),
    op_income_m        = round(total_rev - op_exp, 2),
    arpu_monthly       = arpu
  )
}

fin_df <- generate_financial_data()

# ── 1B. Article-Level Editorial Data (simulated, N = 5,000 articles) ──────────
#
# Simulates NYT article metadata with sentiment scores, section info,
# engagement metrics and publication timestamp.
#

generate_editorial_data <- function(n = 5000) {

  sections <- c("Politics", "Technology", "Business", "Culture",
                "Science", "Sports", "Opinion", "World")

  section_weights <- c(0.22, 0.15, 0.18, 0.10, 0.10, 0.08, 0.10, 0.07)

  pub_dates <- sample(
    seq(as.Date("2019-01-01"), as.Date("2023-12-31"), by = "day"),
    n, replace = TRUE
  )

  section_vec <- sample(sections, n, replace = TRUE, prob = section_weights)

  # Sentiment: Politics/Opinion skew negative; Tech/Science skew positive
  sentiment_means <- c(
    Politics = -0.15, Technology = 0.25, Business = 0.05,
    Culture  = 0.20,  Science    = 0.30, Sports    = 0.35,
    Opinion  = -0.10, World      = -0.05
  )

  sentiment_score <- mapply(
    function(sec) rnorm(1, sentiment_means[sec], 0.35),
    section_vec
  )
  sentiment_score <- pmax(pmin(sentiment_score, 1), -1)

  # Engagement metrics
  page_views   <- round(rlnorm(n, meanlog = 8.5, sdlog = 1.2))
  time_on_page <- round(rnorm(n, 180, 60), 0)  # seconds
  time_on_page <- pmax(time_on_page, 10)
  shares       <- round(page_views * runif(n, 0.001, 0.05))
  comments     <- round(page_views * runif(n, 0.0005, 0.02))

  # Subscription conversions attributed to article
  conversion_prob <- plogis(
    -5 +
      0.8  * sentiment_score +
      0.3  * (section_vec == "Technology") +
      0.2  * (section_vec == "Science") +
      1.2  * log(page_views / 1000) +
      0.01 * (time_on_page / 60)
  )
  conversions <- rbinom(n, size = 1, prob = conversion_prob)

  tibble(
    article_id      = glue("NYT-{sprintf('%05d', seq_len(n))}"),
    pub_date        = pub_dates,
    year            = year(pub_dates),
    month           = month(pub_dates),
    quarter         = quarter(pub_dates),
    section         = section_vec,
    sentiment_score = round(sentiment_score, 4),
    sentiment_label = case_when(
      sentiment_score >  0.15 ~ "Positive",
      sentiment_score < -0.15 ~ "Negative",
      TRUE                    ~ "Neutral"
    ),
    page_views      = page_views,
    time_on_page_s  = time_on_page,
    shares          = shares,
    comments        = comments,
    converted       = conversions
  )
}

art_df <- generate_editorial_data(5000)

log_info("Data generated: {nrow(fin_df)} quarters, {nrow(art_df)} articles")


################################################################################
# MODULE 2 — FEATURE ENGINEERING
################################################################################

log_info("MODULE 2: Feature engineering")

# ── 2A. Financial Features ─────────────────────────────────────────────────────

fin_features <- fin_df %>%
  arrange(date) %>%
  mutate(
    # YoY growth rates
    dig_sub_rev_yoy  = (dig_sub_rev_m / lag(dig_sub_rev_m, 4) - 1) * 100,
    dig_subs_yoy     = (dig_subs_k    / lag(dig_subs_k, 4)    - 1) * 100,
    print_adv_yoy    = (print_adv_m   / lag(print_adv_m, 4)   - 1) * 100,

    # QoQ growth
    dig_sub_rev_qoq  = (dig_sub_rev_m / lag(dig_sub_rev_m, 1) - 1) * 100,

    # Revenue mix
    dig_share        = dig_sub_rev_m / total_rev_m * 100,
    print_share      = print_adv_m   / total_rev_m * 100,

    # Rolling averages
    dig_sub_rev_4qma = rollmean(dig_sub_rev_m, k = 4, fill = NA, align = "right"),

    # Operating margin
    op_margin        = op_income_m / total_rev_m * 100,

    # ARPU trend
    arpu_qoq         = (arpu_monthly / lag(arpu_monthly, 1) - 1) * 100,

    # Time features
    is_q4            = as.integer(quarter == 4),
    trend_idx        = row_number()
  )

# ── 2B. Editorial Aggregated Features (quarterly) ──────────────────────────────

editorial_quarterly <- art_df %>%
  group_by(year, quarter) %>%
  summarise(
    n_articles         = n(),
    avg_sentiment      = mean(sentiment_score),
    pct_positive       = mean(sentiment_label == "Positive") * 100,
    pct_negative       = mean(sentiment_label == "Negative") * 100,
    total_page_views   = sum(page_views),
    avg_time_on_page   = mean(time_on_page_s),
    total_conversions  = sum(converted),
    conversion_rate    = mean(converted) * 100,
    .groups = "drop"
  ) %>%
  mutate(date = as.Date(glue("{year}-{(quarter - 1) * 3 + 1}-01")))

# ── 2C. Merged Master Dataset ──────────────────────────────────────────────────

master_df <- fin_features %>%
  left_join(editorial_quarterly, by = c("year", "quarter", "date")) %>%
  filter(year >= 2019)   # editorial data starts 2019

log_info("Master dataset: {nrow(master_df)} rows, {ncol(master_df)} features")


################################################################################
# MODULE 3 — EXPLORATORY DATA ANALYSIS
################################################################################

log_info("MODULE 3: Generating EDA visualizations")

# ── Theme ──────────────────────────────────────────────────────────────────────

theme_nyt <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      text              = element_text(family = "sans", color = NYT_BLACK),
      plot.title        = element_text(size = base_size + 4, face = "bold",
                                       margin = margin(b = 6)),
      plot.subtitle     = element_text(size = base_size, color = "#555555",
                                       margin = margin(b = 10)),
      plot.caption      = element_text(size = base_size - 2, color = "#888888",
                                       hjust = 0),
      panel.grid.major  = element_line(color = "#e8e8e8", linewidth = 0.4),
      panel.grid.minor  = element_blank(),
      axis.title        = element_text(size = base_size - 1, color = "#444444"),
      axis.text         = element_text(size = base_size - 1),
      legend.position   = "bottom",
      legend.title      = element_text(face = "bold"),
      strip.text        = element_text(face = "bold"),
      plot.background   = element_rect(fill = "white", color = NA),
      panel.background  = element_rect(fill = "white", color = NA)
    )
}

# ── P1: Digital Subscription Revenue vs Subscriber Growth ─────────────────────

p1 <- fin_df %>%
  ggplot(aes(x = date)) +
  geom_area(aes(y = dig_sub_rev_m), fill = NYT_BLUE, alpha = 0.15) +
  geom_line(aes(y = dig_sub_rev_m), color = NYT_BLUE, linewidth = 1.2) +
  geom_point(aes(y = dig_sub_rev_m), color = NYT_BLUE, size = 1.5) +
  scale_y_continuous(labels = dollar_format(suffix = "M"), name = "Revenue ($M)") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(
    title    = "NYT Digital Subscription Revenue (2014–2023)",
    subtitle = "Quarterly revenue showing accelerated post-COVID growth",
    caption  = "Source: Simulated based on NYT public filings",
    x        = NULL
  ) +
  theme_nyt()

# ── P2: Revenue Mix Over Time ──────────────────────────────────────────────────

rev_long <- fin_df %>%
  select(date, dig_sub_rev_m, print_adv_m, dig_adv_m, other_rev_m) %>%
  pivot_longer(-date,
               names_to  = "revenue_type",
               values_to = "revenue_m") %>%
  mutate(revenue_type = recode(revenue_type,
    dig_sub_rev_m = "Digital Subscriptions",
    print_adv_m   = "Print Advertising",
    dig_adv_m     = "Digital Advertising",
    other_rev_m   = "Other (Games, Cooking, Wirecutter)"
  ))

p2 <- rev_long %>%
  ggplot(aes(x = date, y = revenue_m, fill = revenue_type)) +
  geom_area(position = "fill", alpha = 0.85) +
  scale_y_continuous(labels = percent_format(), name = "Revenue Share") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_fill_manual(values = c(
    "Digital Subscriptions"              = NYT_BLUE,
    "Print Advertising"                  = "#8B8B8B",
    "Digital Advertising"                = "#4ECDC4",
    "Other (Games, Cooking, Wirecutter)" = "#F7931E"
  )) +
  labs(
    title    = "NYT Revenue Mix Shift (2014–2023)",
    subtitle = "Structural transformation from print advertising to digital subscriptions",
    caption  = "Source: Simulated based on NYT public filings",
    x        = NULL, fill = NULL
  ) +
  theme_nyt()

# ── P3: Sentiment by Section ───────────────────────────────────────────────────

p3 <- art_df %>%
  group_by(section) %>%
  summarise(
    avg_sentiment = mean(sentiment_score),
    se            = sd(sentiment_score) / sqrt(n()),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = reorder(section, avg_sentiment),
             y = avg_sentiment,
             fill = avg_sentiment > 0)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_errorbar(aes(ymin = avg_sentiment - 1.96 * se,
                    ymax = avg_sentiment + 1.96 * se),
                width = 0.2, color = "#333333") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#666666") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = NYT_BLUE, "FALSE" = NYT_ACCENT),
                    guide = "none") +
  scale_y_continuous(limits = c(-0.3, 0.5)) +
  labs(
    title    = "Average Article Sentiment Score by Section",
    subtitle = "Sports and Science articles trend most positive; Politics leans negative",
    caption  = "95% CI shown | N = 5,000 articles (2019–2023)",
    x        = NULL, y        = "Mean Sentiment Score (−1 to +1)"
  ) +
  theme_nyt()

# ── P4: ARPU Trend ─────────────────────────────────────────────────────────────

p4 <- fin_df %>%
  ggplot(aes(x = date, y = arpu_monthly)) +
  geom_line(color = NYT_BLACK, linewidth = 1) +
  geom_smooth(method = "loess", se = TRUE, color = NYT_BLUE,
              fill = NYT_BLUE, alpha = 0.15, linewidth = 0.7) +
  scale_y_continuous(labels = dollar_format(), name = "Monthly ARPU ($)") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(
    title    = "Monthly Average Revenue Per User (ARPU)",
    subtitle = "LOESS smoothed trend with 95% confidence band",
    caption  = "Digital subscribers only",
    x        = NULL
  ) +
  theme_nyt()

# ── Compose EDA Panel ──────────────────────────────────────────────────────────

eda_panel <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title   = "THE NEW YORK TIMES COMPANY — Digital Transformation Analytics",
    caption = glue("Generated {Sys.Date()} | Production R Analytics Pipeline"),
    theme   = theme(
      plot.title = element_text(size = 16, face = "bold", color = NYT_BLACK),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

ggsave("/mnt/user-data/outputs/nyt_eda_panel.png",
       eda_panel, width = 16, height = 12, dpi = 200, bg = "white")

log_info("EDA panel saved → nyt_eda_panel.png")


################################################################################
# MODULE 4 — PREDICTIVE MODELING
################################################################################

log_info("MODULE 4: Training forecasting models")

# ── 4A. ARIMA Model for Quarterly Digital Sub Revenue ─────────────────────────

ts_dig_rev <- ts(
  fin_df$dig_sub_rev_m,
  start     = c(2014, 1),
  frequency = 4
)

arima_fit <- auto.arima(
  ts_dig_rev,
  seasonal      = TRUE,
  stepwise      = FALSE,
  approximation = FALSE,
  trace         = FALSE
)

log_info("ARIMA model: {paste(arima_fit$arma, collapse=',')}")

arima_forecast <- forecast(arima_fit, h = 8)   # 2 years ahead

# ── 4B. XGBoost for Subscription Revenue with Lagged Features ─────────────────

ml_df <- master_df %>%
  drop_na() %>%
  mutate(
    lag1_dig_sub   = lag(dig_sub_rev_m, 1),
    lag2_dig_sub   = lag(dig_sub_rev_m, 2),
    lag4_dig_sub   = lag(dig_sub_rev_m, 4),
    lag1_subs      = lag(dig_subs_k, 1),
    lag1_sentiment = lag(avg_sentiment, 1),
    lag1_conv_rate = lag(conversion_rate, 1)
  ) %>%
  drop_na()

feature_cols <- c(
  "lag1_dig_sub", "lag2_dig_sub", "lag4_dig_sub",
  "lag1_subs", "lag1_sentiment", "lag1_conv_rate",
  "arpu_monthly", "is_q4", "trend_idx",
  "pct_positive", "total_page_views"
)

X_mat <- as.matrix(ml_df[, feature_cols])
y_vec <- ml_df$dig_sub_rev_m

# Time-series cross-validation: train on first 80%, test on last 20%
n_train  <- floor(0.80 * nrow(ml_df))
X_train  <- X_mat[1:n_train, ]
y_train  <- y_vec[1:n_train]
X_test   <- X_mat[(n_train + 1):nrow(ml_df), ]
y_test   <- y_vec[(n_train + 1):nrow(ml_df)]

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest  <- xgb.DMatrix(X_test,  label = y_test)

xgb_params <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  nthread          = 1
)

xgb_fit <- xgb.train(
  params    = xgb_params,
  data      = dtrain,
  nrounds   = 300,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 30,
  verbose   = 0
)

xgb_preds <- predict(xgb_fit, dtest)

# ── 4C. Simple Ensemble: Average ARIMA + XGBoost ──────────────────────────────

# ARIMA predictions for test window
arima_test_preds <- as.numeric(
  forecast(arima_fit, h = n_train + length(y_test))$mean
)[(n_train + 1):(n_train + length(y_test))]

ensemble_preds <- (xgb_preds + arima_test_preds) / 2


################################################################################
# MODULE 5 — MODEL EVALUATION
################################################################################

log_info("MODULE 5: Model evaluation")

evaluate_model <- function(actual, predicted, model_name) {
  tibble(
    Model = model_name,
    RMSE  = round(rmse(actual, predicted), 3),
    MAE   = round(mae(actual, predicted),  3),
    MAPE  = round(mape(actual, predicted) * 100, 2),
    R2    = round(cor(actual, predicted)^2, 4)
  )
}

eval_results <- bind_rows(
  evaluate_model(y_test, arima_test_preds, "ARIMA (Seasonal)"),
  evaluate_model(y_test, xgb_preds,        "XGBoost"),
  evaluate_model(y_test, ensemble_preds,   "Ensemble (ARIMA + XGBoost)")
)

log_info("Model evaluation results:")
print(eval_results)

# ── Feature Importance Plot ────────────────────────────────────────────────────

importance_df <- xgb.importance(
  feature_names = feature_cols,
  model         = xgb_fit
) %>%
  as_tibble() %>%
  slice_max(Gain, n = 10) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))

p_imp <- importance_df %>%
  ggplot(aes(x = Feature, y = Gain, fill = Gain)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#d6e4f0", high = NYT_BLUE) +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title    = "XGBoost Feature Importance",
    subtitle = "Top 10 features by information gain",
    x        = NULL, y = "Gain (%)"
  ) +
  theme_nyt()

# ── Forecast Plot ──────────────────────────────────────────────────────────────

forecast_df <- tibble(
  date         = seq(as.Date("2024-01-01"), by = "quarter", length.out = 8),
  lower_95     = as.numeric(arima_forecast$lower[, "95%"]),
  upper_95     = as.numeric(arima_forecast$upper[, "95%"]),
  lower_80     = as.numeric(arima_forecast$lower[, "80%"]),
  upper_80     = as.numeric(arima_forecast$upper[, "80%"]),
  mean_forecast = as.numeric(arima_forecast$mean)
)

p_forecast <- ggplot() +
  geom_line(data = fin_df, aes(x = date, y = dig_sub_rev_m),
            color = NYT_BLACK, linewidth = 0.9) +
  geom_ribbon(data = forecast_df,
              aes(x = date, ymin = lower_95, ymax = upper_95),
              fill = NYT_BLUE, alpha = 0.15) +
  geom_ribbon(data = forecast_df,
              aes(x = date, ymin = lower_80, ymax = upper_80),
              fill = NYT_BLUE, alpha = 0.25) +
  geom_line(data = forecast_df,
            aes(x = date, y = mean_forecast),
            color = NYT_BLUE, linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2024-01-01"),
             linetype = "dotted", color = NYT_ACCENT, linewidth = 0.8) +
  scale_y_continuous(labels = dollar_format(suffix = "M"),
                     name = "Digital Subscription Revenue ($M)") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(
    title    = "NYT Digital Subscription Revenue Forecast (2024–2025)",
    subtitle = "ARIMA seasonal model | 80% and 95% prediction intervals",
    caption  = "Dashed vertical line = forecast origin",
    x        = NULL
  ) +
  theme_nyt()

modeling_panel <- (p_forecast | p_imp) +
  plot_annotation(
    title = "THE NEW YORK TIMES COMPANY — Forecast & Feature Analysis",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

ggsave("/mnt/user-data/outputs/nyt_modeling_panel.png",
       modeling_panel, width = 16, height = 7, dpi = 200, bg = "white")

log_info("Modeling panel saved → nyt_modeling_panel.png")

# ── Save evaluation table ──────────────────────────────────────────────────────
write_csv(eval_results, "/mnt/user-data/outputs/nyt_model_evaluation.csv")
log_info("Model eval saved → nyt_model_evaluation.csv")


################################################################################
# MODULE 6 — SHINY DASHBOARD
################################################################################

log_info("MODULE 6: Building Shiny dashboard")

# ── Pre-process data for dashboard ────────────────────────────────────────────

# Ensure fin_df and art_df are accessible inside server
dash_fin  <- fin_df
dash_art  <- art_df
dash_eval <- eval_results
dash_fore <- forecast_df

# ──────────────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────────────

ui <- dashboardPage(
  skin = "black",

  dashboardHeader(
    title = tags$span(
      tags$b("NYT", style = "font-family: Georgia, serif; color: white;"),
      " Analytics"
    )
  ),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Revenue Overview",  tabName = "revenue",   icon = icon("chart-line")),
      menuItem("Editorial Insights",tabName = "editorial", icon = icon("newspaper")),
      menuItem("Forecasting",       tabName = "forecast",  icon = icon("binoculars")),
      menuItem("Model Evaluation",  tabName = "models",    icon = icon("table"))
    ),
    hr(),
    sliderInput("year_range", "Year Range",
                min = 2014, max = 2023, value = c(2014, 2023), sep = ""),
    selectInput("section_filter", "Article Section",
                choices = c("All", unique(art_df$section)), selected = "All")
  ),

  dashboardBody(

    tags$head(
      tags$style(HTML(glue("
        .content-wrapper, .main-sidebar {{ background-color: #f9f9f7; }}
        .box {{ border-top: 3px solid {NYT_BLUE}; border-radius: 2px; }}
        .skin-black .main-header .logo {{ background-color: {NYT_BLACK}; }}
        .skin-black .main-header .navbar {{ background-color: {NYT_BLACK}; }}
        .value-box .inner h3 {{ font-size: 28px; }}
        h4 {{ font-family: Georgia, serif; }}
      ")))
    ),

    tabItems(

      # ── TAB 1: Revenue Overview ──────────────────────────────────────────────
      tabItem(tabName = "revenue",

        fluidRow(
          valueBoxOutput("vbox_rev",   width = 3),
          valueBoxOutput("vbox_subs",  width = 3),
          valueBoxOutput("vbox_arpu",  width = 3),
          valueBoxOutput("vbox_share", width = 3)
        ),

        fluidRow(
          box(title = "Digital Subscription Revenue", width = 8, solidHeader = TRUE,
              plotlyOutput("plot_rev_ts", height = "320px")),
          box(title = "Revenue Mix (Latest Quarter)", width = 4, solidHeader = TRUE,
              plotlyOutput("plot_rev_pie", height = "320px"))
        ),

        fluidRow(
          box(title = "YoY Revenue Growth (%)", width = 6,
              plotlyOutput("plot_yoy", height = "260px")),
          box(title = "Operating Margin (%)", width = 6,
              plotlyOutput("plot_margin", height = "260px"))
        )
      ),

      # ── TAB 2: Editorial Insights ────────────────────────────────────────────
      tabItem(tabName = "editorial",

        fluidRow(
          box(title = "Sentiment Distribution by Section", width = 7,
              plotlyOutput("plot_sentiment", height = "360px")),
          box(title = "Conversion Rate by Sentiment Label", width = 5,
              plotlyOutput("plot_conversion", height = "360px"))
        ),

        fluidRow(
          box(title = "Top Articles by Page Views", width = 12,
              DTOutput("tbl_articles"))
        )
      ),

      # ── TAB 3: Forecasting ───────────────────────────────────────────────────
      tabItem(tabName = "forecast",

        fluidRow(
          box(title = "8-Quarter Revenue Forecast (ARIMA + 95% CI)",
              width = 12, solidHeader = TRUE,
              plotlyOutput("plot_forecast", height = "420px"))
        ),

        fluidRow(
          box(title = "Forecast Values", width = 6,
              DTOutput("tbl_forecast")),
          box(title = "Forecast Assumptions", width = 6,
              tags$ul(
                tags$li("Model: ARIMA with seasonal differencing"),
                tags$li("Forecast horizon: 8 quarters (2024 Q1 – 2025 Q4)"),
                tags$li("Assumes no major structural shocks"),
                tags$li("80% and 95% prediction intervals shown"),
                tags$li("Historical growth CAGR ≈ 22% (2014–2023)")
              ))
        )
      ),

      # ── TAB 4: Model Evaluation ──────────────────────────────────────────────
      tabItem(tabName = "models",

        fluidRow(
          box(title = "Model Performance Comparison", width = 8,
              plotlyOutput("plot_eval", height = "300px")),
          box(title = "Evaluation Metrics Table", width = 4,
              DTOutput("tbl_eval"))
        ),

        fluidRow(
          box(title = "XGBoost Feature Importance", width = 12,
              plotlyOutput("plot_importance", height = "320px"))
        )
      )
    )
  )
)

# ──────────────────────────────────────────────────────────────────────────────
#  SERVER
# ──────────────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # ── Reactive filtered data ──────────────────────────────────────────────────

  fin_filtered <- reactive({
    dash_fin %>%
      filter(year >= input$year_range[1], year <= input$year_range[2])
  })

  art_filtered <- reactive({
    d <- dash_art %>%
      filter(year >= input$year_range[1], year <= input$year_range[2])
    if (input$section_filter != "All") {
      d <- filter(d, section == input$section_filter)
    }
    d
  })

  latest <- reactive({
    fin_filtered() %>% slice_max(date, n = 1)
  })

  # ── Value Boxes ─────────────────────────────────────────────────────────────

  output$vbox_rev <- renderValueBox({
    valueBox(
      dollar(latest()$dig_sub_rev_m, suffix = "M"),
      "Digital Sub Revenue (Latest Q)",
      icon = icon("dollar-sign"), color = "blue"
    )
  })

  output$vbox_subs <- renderValueBox({
    valueBox(
      paste0(comma(latest()$dig_subs_k), "K"),
      "Digital Subscribers",
      icon = icon("users"), color = "black"
    )
  })

  output$vbox_arpu <- renderValueBox({
    valueBox(
      dollar(latest()$arpu_monthly),
      "Monthly ARPU",
      icon = icon("wallet"), color = "navy"
    )
  })

  output$vbox_share <- renderValueBox({
    valueBox(
      percent(latest()$dig_share / 100, accuracy = 0.1),
      "Digital Revenue Share",
      icon = icon("chart-pie"), color = "black"
    )
  })

  # ── Revenue Time Series ──────────────────────────────────────────────────────

  output$plot_rev_ts <- renderPlotly({
    p <- fin_filtered() %>%
      plot_ly(x = ~date, y = ~dig_sub_rev_m, type = "scatter", mode = "lines+markers",
              line = list(color = NYT_BLUE, width = 2),
              marker = list(size = 5, color = NYT_BLUE),
              name = "Digital Sub Rev") %>%
      add_lines(y = ~dig_sub_rev_4qma, line = list(color = NYT_ACCENT, dash = "dash"),
                name = "4Q Moving Avg") %>%
      layout(
        yaxis  = list(title = "Revenue ($M)", tickprefix = "$"),
        xaxis  = list(title = ""),
        legend = list(orientation = "h"),
        hovermode = "x unified",
        plot_bgcolor  = "white",
        paper_bgcolor = "white"
      )
    p
  })

  output$plot_rev_pie <- renderPlotly({
    latest_q <- latest()
    vals <- c(latest_q$dig_sub_rev_m, latest_q$print_adv_m,
              latest_q$dig_adv_m,     latest_q$other_rev_m)
    labs <- c("Digital Subs", "Print Advertising", "Digital Advertising", "Other")
    plot_ly(labels = labs, values = vals, type = "pie",
            marker = list(colors = c(NYT_BLUE, "#8B8B8B", "#4ECDC4", "#F7931E")),
            textinfo = "label+percent",
            hoverinfo = "label+value+percent") %>%
      layout(showlegend = FALSE, paper_bgcolor = "white")
  })

  output$plot_yoy <- renderPlotly({
    fin_filtered() %>%
      drop_na(dig_sub_rev_yoy) %>%
      plot_ly(x = ~date, y = ~dig_sub_rev_yoy, type = "bar",
              marker = list(color = ~ifelse(dig_sub_rev_yoy >= 0, NYT_BLUE, NYT_ACCENT))) %>%
      layout(yaxis = list(title = "YoY Growth (%)", ticksuffix = "%"),
             xaxis = list(title = ""),
             plot_bgcolor = "white", paper_bgcolor = "white")
  })

  output$plot_margin <- renderPlotly({
    fin_filtered() %>%
      drop_na(op_margin) %>%
      plot_ly(x = ~date, y = ~op_margin, type = "scatter", mode = "lines+markers",
              line   = list(color = NYT_BLACK, width = 1.5),
              marker = list(size = 4)) %>%
      add_lines(y = ~0, line = list(color = "red", dash = "dot", width = 1)) %>%
      layout(yaxis = list(title = "Operating Margin (%)", ticksuffix = "%"),
             xaxis = list(title = ""),
             showlegend = FALSE,
             plot_bgcolor = "white", paper_bgcolor = "white")
  })

  # ── Editorial ────────────────────────────────────────────────────────────────

  output$plot_sentiment <- renderPlotly({
    art_filtered() %>%
      plot_ly(x = ~section, y = ~sentiment_score, type = "box",
              color = ~section, colors = "Set2",
              boxpoints = FALSE) %>%
      layout(
        xaxis = list(title = ""),
        yaxis = list(title = "Sentiment Score (−1 to +1)", zeroline = TRUE,
                     zerolinecolor = "#aaa", zerolinewidth = 1),
        showlegend = FALSE,
        plot_bgcolor = "white", paper_bgcolor = "white"
      )
  })

  output$plot_conversion <- renderPlotly({
    art_filtered() %>%
      group_by(sentiment_label) %>%
      summarise(conv_rate = mean(converted) * 100, .groups = "drop") %>%
      plot_ly(x = ~sentiment_label, y = ~conv_rate, type = "bar",
              marker = list(color = c(NYT_BLUE, "#aaa", NYT_ACCENT)),
              text = ~round(conv_rate, 2), textposition = "outside") %>%
      layout(xaxis = list(title = "Sentiment Label"),
             yaxis = list(title = "Conversion Rate (%)", ticksuffix = "%"),
             plot_bgcolor = "white", paper_bgcolor = "white")
  })

  output$tbl_articles <- renderDT({
    art_filtered() %>%
      arrange(desc(page_views)) %>%
      slice_head(n = 50) %>%
      select(article_id, pub_date, section, sentiment_label,
             page_views, shares, comments, converted) %>%
      datatable(
        options = list(pageLength = 8, scrollX = TRUE,
                       dom = "ftip",
                       columnDefs = list(list(className = "dt-center", targets = "_all"))),
        rownames = FALSE,
        class    = "stripe hover compact"
      )
  })

  # ── Forecast ─────────────────────────────────────────────────────────────────

  output$plot_forecast <- renderPlotly({
    historical <- fin_filtered() %>%
      select(date, y = dig_sub_rev_m) %>%
      mutate(type = "Historical")

    plot_ly() %>%
      add_ribbons(data = dash_fore, x = ~date,
                  ymin = ~lower_95, ymax = ~upper_95,
                  fillcolor = "rgba(50, 104, 145, 0.15)",
                  line = list(color = "transparent"),
                  name = "95% CI") %>%
      add_ribbons(data = dash_fore, x = ~date,
                  ymin = ~lower_80, ymax = ~upper_80,
                  fillcolor = "rgba(50, 104, 145, 0.25)",
                  line = list(color = "transparent"),
                  name = "80% CI") %>%
      add_lines(data = historical, x = ~date, y = ~y,
                line = list(color = NYT_BLACK, width = 2), name = "Historical") %>%
      add_lines(data = dash_fore, x = ~date, y = ~mean_forecast,
                line = list(color = NYT_BLUE, width = 2, dash = "dash"),
                name = "Forecast") %>%
      layout(
        xaxis     = list(title = ""),
        yaxis     = list(title = "Digital Sub Revenue ($M)", tickprefix = "$"),
        hovermode = "x unified",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        legend = list(orientation = "h")
      )
  })

  output$tbl_forecast <- renderDT({
    dash_fore %>%
      mutate(across(where(is.numeric), ~ round(.x, 1))) %>%
      select(date, mean_forecast, lower_80, upper_80, lower_95, upper_95) %>%
      rename(
        Date = date,
        `Forecast ($M)` = mean_forecast,
        `Lower 80%`     = lower_80,
        `Upper 80%`     = upper_80,
        `Lower 95%`     = lower_95,
        `Upper 95%`     = upper_95
      ) %>%
      datatable(options = list(pageLength = 8, dom = "t"), rownames = FALSE)
  })

  # ── Model Evaluation ─────────────────────────────────────────────────────────

  output$plot_eval <- renderPlotly({
    eval_long <- dash_eval %>%
      pivot_longer(c(RMSE, MAE, MAPE), names_to = "Metric", values_to = "Value")

    plot_ly(eval_long, x = ~Model, y = ~Value, color = ~Metric,
            type = "bar", barmode = "group",
            colors = c(NYT_BLUE, "#4ECDC4", NYT_ACCENT)) %>%
      layout(
        xaxis = list(title = ""),
        yaxis = list(title = "Error"),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        legend = list(orientation = "h")
      )
  })

  output$tbl_eval <- renderDT({
    dash_eval %>%
      datatable(
        options = list(pageLength = 5, dom = "t"),
        rownames = FALSE, class = "stripe compact"
      )
  })

  output$plot_importance <- renderPlotly({
    importance_df %>%
      plot_ly(x = ~Gain, y = ~Feature, type = "bar", orientation = "h",
              marker = list(color = NYT_BLUE)) %>%
      layout(
        xaxis = list(title = "Gain (Information Gain)", tickformat = ".1%"),
        yaxis = list(title = "", categoryorder = "total ascending"),
        plot_bgcolor  = "white",
        paper_bgcolor = "white"
      )
  })
}

# ── Run Application ────────────────────────────────────────────────────────────

log_info("Shiny dashboard defined — run shinyApp(ui, server) to launch")

# Uncomment to launch interactively:
# shinyApp(ui = ui, server = server)


################################################################################
# EXPORT SUMMARY REPORT
################################################################################

log_info("Exporting summary report")

summary_lines <- c(
  "=" ,
  "NYT ANALYTICS PIPELINE — EXECUTION SUMMARY",
  "=" ,
  glue("Run timestamp        : {Sys.time()}"),
  glue("Financial quarters   : {nrow(fin_df)}  (2014 Q1 – 2023 Q4)"),
  glue("Articles analyzed    : {nrow(art_df)}"),
  glue("Features engineered  : {ncol(master_df)}"),
  "",
  "ARIMA Model Order    :",
  capture.output(print(arima_fit)),
  "",
  "MODEL EVALUATION:",
  capture.output(print(eval_results, n = 10)),
  "",
  "TOP 5 XGBoost Features (by Gain):",
  capture.output(print(slice_max(importance_df, Gain, n = 5), n = 5)),
  "",
  "ARIMA 8-Quarter Forecast (mean $M):",
  capture.output(print(round(as.numeric(arima_forecast$mean), 2)))
)

writeLines(summary_lines, "/mnt/user-data/outputs/nyt_summary_report.txt")

log_info("Pipeline complete. Outputs:")
log_info("  • nyt_analytics.R          — full source code")
log_info("  • nyt_eda_panel.png        — 4-panel EDA visualization")
log_info("  • nyt_modeling_panel.png   — forecast + feature importance")
log_info("  • nyt_model_evaluation.csv — RMSE / MAE / MAPE table")
log_info("  • nyt_summary_report.txt   — execution summary")
