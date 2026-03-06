# 📰 NYT Company — Digital Subscription Analytics Pipeline

> A production-grade R pipeline that models the digital transformation of **The New York Times Company** — from revenue forecasting and editorial sentiment analysis to an interactive Shiny executive dashboard.

---

## 📌 Problem Statement

The New York Times Company has undergone one of the most successful media transformations in history — shifting from a print-advertising-dependent publisher to a digital subscription powerhouse with nearly 10 million subscribers by 2023.

This project answers the key business question:

> **What are the primary drivers of NYT's digital subscription revenue, and how can we forecast its trajectory while integrating editorial sentiment signals?**

---

## 🗂️ Project Structure

```
nyt-analytics/
│
├── nyt_analytics.R            # Full production pipeline (all 6 modules)
│
├── outputs/
│   ├── nyt_eda_panel.png      # 4-panel EDA visualization
│   ├── nyt_modeling_panel.png # Forecast + feature importance plots
│   ├── nyt_model_evaluation.csv
│   └── nyt_summary_report.txt
│
└── README.md
```

---

## 🔬 Pipeline Modules

| Module | Description |
|--------|-------------|
| **1 — Data Ingestion** | Simulates 40 quarters of NYT financials (2014–2023) + 5,000 article-level records with section, sentiment, engagement, and conversion labels |
| **2 — Feature Engineering** | YoY/QoQ growth rates, rolling 4Q moving averages, digital revenue share, ARPU trends, lagged ML features, quarterly editorial aggregations |
| **3 — EDA** | Publication-quality 4-panel ggplot2 figure: revenue growth, revenue mix shift, sentiment by section, ARPU trend |
| **4 — Predictive Modeling** | Seasonal ARIMA (`auto.arima`) + XGBoost with editorial features, combined into a simple ensemble |
| **5 — Model Evaluation** | RMSE, MAE, MAPE, R² across all three models; XGBoost feature importance |
| **6 — Shiny Dashboard** | 4-tab interactive dashboard: Revenue Overview, Editorial Insights, Forecasting, Model Evaluation |

---

## 📊 Visualizations

### EDA Panel
Four-panel analysis covering:
- Digital subscription revenue trend (2014–2023)
- Revenue mix shift from print advertising → digital subscriptions
- Article sentiment distribution by editorial section
- Monthly ARPU with LOESS smoothing

### Modeling Panel
- ARIMA 8-quarter forecast with 80% and 95% prediction intervals
- XGBoost top-10 feature importance by information gain

---

## 🤖 Models

### Seasonal ARIMA
- Fitted via `forecast::auto.arima` with full grid search (`stepwise = FALSE`, `approximation = FALSE`)
- Captures quarterly seasonality in digital subscription revenue
- Forecasts 8 quarters ahead (2024 Q1 – 2025 Q4)

### XGBoost
- Features: lagged revenue, subscriber count, ARPU, editorial sentiment, conversion rate, page views, Q4 dummy, trend index
- Hyperparameters: `eta = 0.05`, `max_depth = 4`, `subsample = 0.8`, `colsample_bytree = 0.8`
- Early stopping with 30-round patience on held-out validation set
- Time-series cross-validation: 80% train / 20% test (no data leakage)

### Ensemble
- Simple average of ARIMA and XGBoost predictions
- Consistently outperforms either model alone on MAPE

---

## 🧰 Tech Stack

| Category | Packages |
|----------|----------|
| Data Wrangling | `tidyverse`, `lubridate`, `janitor`, `zoo` |
| Modeling | `forecast`, `xgboost`, `caret`, `Metrics` |
| Visualization | `ggplot2`, `patchwork`, `ggthemes`, `viridis`, `scales` |
| Dashboard | `shiny`, `shinydashboard`, `plotly`, `DT` |
| Utilities | `glue` |

---

## ⚙️ Installation

### Prerequisites
- R ≥ 4.2.0
- RStudio (recommended)

### Install Required Packages

```r
install.packages(c(
  "tidyverse", "lubridate", "scales", "janitor", "zoo", "glue",
  "forecast", "xgboost", "caret", "Metrics",
  "ggplot2", "patchwork", "ggthemes", "viridis",
  "shiny", "shinydashboard", "plotly", "DT"
))
```

---

## 🚀 Usage

### Run Full Pipeline

```r
source("nyt_analytics.R")
```

This will:
1. Generate all datasets
2. Engineer features
3. Save EDA and modeling plots to `outputs/`
4. Train ARIMA + XGBoost + Ensemble
5. Print evaluation metrics to console
6. Define the Shiny dashboard (ready to launch)

### Launch Shiny Dashboard

At the bottom of the script, uncomment:

```r
shinyApp(ui = ui, server = server)
```

Or run after sourcing:

```r
shinyApp(ui = ui, server = server)
```

The dashboard will open at `http://127.0.0.1:PORT` in your browser.

---

## 📈 Key Findings

- **Lagged subscriber count** is the strongest predictor of digital subscription revenue (highest XGBoost gain)
- **Technology and Science sections** produce the most positive sentiment and highest subscription conversion rates
- **Politics and Opinion** sections skew negative in sentiment, yet drive high page view volume
- **Q4 seasonality** consistently boosts revenue ~8–12% vs. prior quarter
- **ARPU** has remained relatively flat despite subscriber growth, suggesting pricing headroom
- **Ensemble model** achieves the lowest MAPE, outperforming both standalone models

---

## 🗄️ Data Sources

All financial data is **simulated** based on publicly available NYT Company quarterly earnings reports and investor presentations. Article data is synthetically generated to reflect realistic editorial patterns. This project is intended for educational and portfolio purposes only.

Reference sources:
- [NYT Company Investor Relations](https://investors.nytco.com)
- [NYT Q4 2023 Earnings Report](https://investors.nytco.com/financials/quarterly-earnings)

---

## 🔭 Future Work

- [ ] Integrate real NYT Article Search API for live sentiment scoring
- [ ] Add LSTM/Prophet comparison for long-horizon forecasting
- [ ] Build subscriber churn prediction model
- [ ] Incorporate macro signals (unemployment, CPI) as external regressors in ARIMAX
- [ ] Deploy Shiny dashboard to [shinyapps.io](https://www.shinyapps.io)
- [ ] Add unit tests with `testthat`

---

## 👩‍💻 Author

**Pooja Venugopal Baskaran**    
[LinkedIn](https://linkedin.com/in/pooja-venugopal-baskaran)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
