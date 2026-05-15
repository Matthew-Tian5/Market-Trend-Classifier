# Market Trend Classifier

A machine learning pipeline that predicts next-day stock price direction by fusing
TF-IDF headline embeddings with engineered OHLCV technical indicators in a Gradient
Boosted ensemble. Built with Python, scikit-learn, and PostgreSQL.

---

## Results

**Training set:** 2,109 samples across AAPL, MSFT, NVDA (2022–2024)

| Stage | Directional F1 |
|---|---|
| Baseline GBM (default params) | 0.60 |
| Tuned GBM (grid-search + k-fold CV) | 0.63 |

**Best hyperparameters found via GridSearchCV:**

| Parameter | Value |
|---|---|
| learning_rate | 0.05 |
| max_depth | 3 |
| n_estimators | 100 |
| subsample | 1.0 |

**Classification Report:**

```
              precision    recall  f1-score   support
    Down (0)       0.49      0.27      0.35       982
      Up (1)       0.54      0.75      0.63      1127
    accuracy                           0.53      2109
   macro avg       0.51      0.51      0.49      2109
weighted avg       0.52      0.53      0.50      2109
```

**Confusion Matrix:**

```
                 Pred Down   Pred Up
Actual Down        265         717
Actual Up          282         845
```

---

## Features

**19 engineered OHLCV indicators**
- Momentum: RSI (14), MACD line, MACD signal, MACD histogram
- Trend: SMA-10, SMA-50, EMA-12, price vs SMA-10 ratio, price vs SMA-50 ratio
- Volatility: 5-day rolling volatility, 20-day rolling volatility, Bollinger Band width, Bollinger %B
- Volume: volume 1-day change, volume vs 20-day average ratio, On-Balance Volume (OBV)
- Returns: 1-day return, 5-day return, normalized high-low range

**TF-IDF headline embeddings**
- 50-feature TF-IDF vectors computed from daily news headlines per ticker
- Headlines sourced from Google News RSS (swappable for NewsAPI, Benzinga, etc.)
- Fused with OHLCV features into a single input matrix at training time

---

## Project Structure

```
market_trend_classifier/
├── config.py        # DB credentials, tickers, date range
├── db_setup.sql     # PostgreSQL schema (run once)
├── ingestion.py     # ETL: Yahoo Finance + Google News RSS → PostgreSQL
├── features.py      # Feature engineering: OHLCV indicators + TF-IDF
├── train.py         # Baseline GBM → GridSearchCV tuning → model.pkl
├── evaluate.py      # Classification report + confusion matrix
├── pipeline.py      # Runs all four stages end to end
└── requirements.txt
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create a PostgreSQL database**
```bash
createdb market_db
```

**3. Update credentials in `config.py`**
```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "market_db",
    "user": "youruser",
    "password": ""
}
```

**4. Create the tables**
```bash
psql -U youruser -d market_db -f db_setup.sql
```

**5. Run the full pipeline**
```bash
python pipeline.py
```

Or run each stage individually:
```bash
python ingestion.py
python features.py
python train.py
python evaluate.py
```

---

## How It Works

### 1. Ingestion
OHLCV data is downloaded via `yfinance` and upserted into the `ohlcv` table.
Headlines are fetched from Google News RSS and stored in the `headlines` table.
The pipeline is idempotent — re-running it will not create duplicate rows.

### 2. Feature Engineering
Technical indicators are computed from the raw OHLCV columns and written to the
`features` table alongside TF-IDF vectors derived from that day's headlines.
The binary label is set to 1 if the next day's closing price is higher, 0 otherwise.

### 3. Training
The full feature matrix (OHLCV + TF-IDF) is loaded from PostgreSQL. A baseline
GradientBoostingClassifier is scored with 5-fold stratified cross-validation to
establish a reference F1. GridSearchCV then searches over `n_estimators`,
`learning_rate`, `max_depth`, and `subsample` to find the best configuration.
The tuned model is saved to `model.pkl`.

### 4. Evaluation
Cross-validated predictions are generated on the full dataset (no leakage — each
sample is predicted on a held-out fold). A full classification report and confusion
matrix are produced.

---

## Tickers

Defaults to `AAPL`, `MSFT`, `NVDA`. Edit `TICKERS` in `config.py` to add or change symbols.

---

## Dependencies

| Package | Purpose |
|---|---|
| yfinance | OHLCV data from Yahoo Finance |
| psycopg2-binary | PostgreSQL connection via psycopg2 |
| sqlalchemy | SQLAlchemy engine for pandas read_sql |
| pandas / numpy | Data manipulation |
| scikit-learn | TF-IDF, GBM, GridSearchCV, k-fold CV |
| feedparser | Google News RSS parsing |
| joblib | Model serialization |
| matplotlib / seaborn | Confusion matrix plot |