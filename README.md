# Financial News Sentiment vs. NIFTY50 Analysis

A data analysis and machine-learning project investigating the relationship between financial news sentiment and NIFTY50 market direction.

The project is divided into two stages:

- **Part A:** Rule-based financial news sentiment analysis and same-day NIFTY50 movement analysis.
- **Part B:** Time-aware machine-learning models for predicting the **next trading day's** NIFTY50 direction.

> **Important:** This project is for academic analysis and is not financial advice or a trading recommendation.

## Project Overview

The central question of this project is:

> **Can financial news sentiment provide useful information about NIFTY50 market movement?**

Daily financial news and NIFTY50 market data from **February 2025 to August 2025** were cleaned, aligned by trading date, and analyzed.

The project compares a simple rule-based sentiment approach with several machine-learning models and an illustrative backtest.

## Project Structure

```text
Financial-News-NIFTY50-Analysis/
│
├── data/
│   ├── news.csv
│   └── nifty50.csv
│
├── notebooks/
│   ├── 01_analysis.ipynb
│   └── 02_ml_pipeline.ipynb
│
├── plots/
│   ├── barplot.png
│   ├── histogram.png
│   └── scatterplot.png
│
├── results/
│   ├── final_project_dataset.csv
│   ├── grouped_news.csv
│   ├── part_b_backtest_results.csv
│   ├── part_b_feature_importance.csv
│   ├── part_b_ml_dataset.csv
│   └── part_b_model_comparison.csv
│
├── documentation/
│   ├── Part_A_Project_Log.docx
│   ├── Part_B_Project_Log.docx
│   └── Project_Summary.docx
│
├── requirements.txt
└── README.md
```

## Part A — Rule-Based Sentiment Analysis

### Data Preparation

The news and NIFTY50 datasets were inspected and cleaned before analysis.

The news data contained multiple article records per date. News articles were grouped by trading date and combined before being merged with the NIFTY50 market data.

The NIFTY50 data was cleaned by standardizing column names, converting dates to datetime format, converting numerical fields to appropriate numeric types, and aligning trading dates with the news dataset.

The resulting merged dataset contained **132 observations**.

### Sentiment Scoring

A manually curated financial-news vocabulary was used to calculate a daily sentiment score.

```text
Sentiment Score =
Number of positive-word matches
−
Number of negative-word matches
```

Positive scores indicate more positive financial language, while negative scores indicate more negative financial language.

### Market Target

For Part A, market direction was based on the same day's movement:

```text
movement = close − open
```

The target was:

```text
1 → Up
0 → Down
```

The final target distribution was:

| Target | Count |
|---|---:|
| Down (0) | 67 |
| Up (1) | 65 |

The majority-class baseline was **50.76%**.

### Part A Result

A simple sentiment-based prediction rule was used:

```text
Sentiment > 0 → UP
Sentiment < 0 → DOWN
Sentiment = 0 → NEUTRAL
```

For binary evaluation:

```text
Sentiment > 0 → 1
Sentiment ≤ 0 → 0
```

The resulting prediction accuracy was **59.85%**.

This indicates some same-day association between the rule-based sentiment score and market direction in this dataset, but it should not be interpreted as evidence of reliable market prediction.

## Part B — Machine Learning Pipeline

Part B changes the problem from same-day analysis to **next-day prediction**.

### Prediction Target

The target was shifted by one trading day:

```python
merged['next_day_target'] = merged['target'].shift(-1)
```

Therefore:

```text
Today's features → Tomorrow's market direction
```

After removing the final observation without a known next-day target and rows with incomplete engineered features, the ML dataset contained **125 observations**.

The target distribution was:

| Target | Count |
|---|---:|
| Down (0) | 66 |
| Up (1) | 65 |

### Features

Eight features were used:

| Feature | Description |
|---|---|
| `sentiment_score` | Daily rule-based news sentiment |
| `sentiment_ma3` | 3-day moving average of sentiment |
| `sentiment_ma7` | 7-day moving average of sentiment |
| `sentiment_std3` | 3-day rolling sentiment standard deviation |
| `movement_lag1` | Previous trading day's market movement |
| `movement_lag2` | Market movement from two trading days earlier |
| `volume_change` | Percentage change in shares traded |
| `high_low_spread` | Intraday high-low percentage spread |

Rolling and lagged features naturally produced missing values at the beginning of the dataset. These incomplete rows were removed before model training.

## Time-Aware Train/Test Split

Because this is financial time-series data, the observations were **not randomly shuffled**.

An 80/20 chronological split was used:

| Split | Observations | Period |
|---|---:|---|
| Training | 100 | 2025-02-17 → 2025-07-15 |
| Test | 25 | 2025-07-16 → 2025-08-20 |

The test period therefore represents a later period than the training data.

## Models

The following models were evaluated:

1. Majority-class baseline
2. Logistic Regression
3. Random Forest
4. Tuned Random Forest
5. XGBoost

Logistic Regression used feature standardization with `StandardScaler`.

Random Forest hyperparameters were tuned using `TimeSeriesSplit` and `GridSearchCV` rather than shuffled cross-validation.

### Best Random Forest Parameters

```text
n_estimators = 100
max_depth = 3
min_samples_leaf = 20
```

Best time-series cross-validation accuracy:

**51.25%**

## Model Results

The held-out test accuracies were:

| Model | Test Accuracy |
|---|---:|
| Majority Baseline | **56.0%** |
| Logistic Regression | 44.0% |
| Random Forest | 52.0% |
| Tuned Random Forest | 44.0% |
| XGBoost | 40.0% |

The initial Random Forest was the best-performing machine-learning model at **52.0%**, but it still did not outperform the **56.0% majority baseline**.

## Backtest

An illustrative backtest was performed using the tuned Random Forest.

The strategy was:

```text
Prediction = UP    → Invest in NIFTY50 for the next trading day
Prediction = DOWN → Stay in cash
```

The strategy was compared with NIFTY50 buy-and-hold using a normalized starting value of ₹1.

| Strategy | Final Value | Total Return |
|---|---:|---:|
| Tuned Random Forest Strategy | 0.992338 | -0.7662% |
| NIFTY50 Buy-and-Hold | 0.994911 | -0.5089% |

Both approaches lost value during the test period, with the model strategy slightly underperforming buy-and-hold.

### Backtest limitations

The backtest does **not** account for:

- transaction costs,
- slippage,
- taxes,
- or other real-world trading frictions.

The test period is also very small, so the backtest should be considered **illustrative rather than evidence of a profitable trading strategy**.

## Feature Importance

The tuned Random Forest produced the following feature importance ranking:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `movement_lag1` | 0.182939 |
| 2 | `sentiment_ma7` | 0.174181 |
| 3 | `high_low_spread` | 0.150965 |
| 4 | `sentiment_std3` | 0.140246 |
| 5 | `sentiment_ma3` | 0.113807 |
| 6 | `movement_lag2` | 0.110487 |
| 7 | `sentiment_score` | 0.082579 |
| 8 | `volume_change` | 0.044796 |

The most important individual feature was `movement_lag1`, followed by the 7-day sentiment moving average.

The raw daily `sentiment_score` ranked seventh, suggesting that the model relied more heavily on recent market behaviour and smoothed/aggregated sentiment features than on the raw daily sentiment score.

Feature importance should not be interpreted as causal evidence.

## Key Findings

1. The rule-based sentiment approach achieved **59.85% same-day accuracy** in Part A.
2. Moving to a stricter **next-day prediction** problem reduced predictive performance substantially.
3. None of the tested ML models exceeded the **56.0% majority baseline**.
4. The initial Random Forest was the strongest ML model at **52.0%**.
5. The tuned Random Forest achieved **44.0%** on the held-out test set.
6. The illustrative tuned-Random-Forest strategy returned **−0.7662%**, slightly worse than NIFTY50 buy-and-hold at **−0.5089%**.
7. `movement_lag1` was the most important individual feature in the tuned Random Forest.
8. The results do not provide strong evidence that this small dataset supports reliable next-day NIFTY50 direction prediction.

## Limitations

- The ML dataset contains only **125 observations** after feature engineering.
- The held-out test set contains only **25 observations**, so each prediction has a relatively large effect on accuracy.
- The data covers a limited period and may not represent different market regimes.
- The sentiment score is based on a manually curated vocabulary rather than a more sophisticated NLP model.
- The backtest does not include transaction costs, slippage, or taxes.
- Feature importance from a single fitted Random Forest should not be interpreted as causal.
- The results should not be interpreted as a trading recommendation.

## Reproducibility

The project uses Python and the dependencies listed in `requirements.txt`.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Financial-News-NIFTY50-Analysis
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebooks

Open:

```text
notebooks/01_analysis.ipynb
notebooks/02_ml_pipeline.ipynb
```

The notebooks contain the complete analysis and saved outputs.

## Documentation

Additional project documentation is provided in the `documentation/` directory:

- **Part A Project Log** — detailed record of data preparation and sentiment analysis.
- **Part B Project Log** — detailed record of the machine-learning pipeline, evaluation, backtest, and feature importance.
- **Project Summary** — concise one-page summary of the project and findings.

## Disclaimer

This project was developed for academic and analytical purposes. It does not constitute financial advice, investment advice, or a recommendation to buy or sell securities.
