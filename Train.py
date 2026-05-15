import psycopg2
import pandas as pd
import numpy as np
import json
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from Config import DB_CONFIG, TICKERS
from Features import build_ohlcv_features, make_label, OHLCV_FEATURE_COLS


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    c = DB_CONFIG
    url = f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
    return create_engine(url)


def load_feature_matrix(tickers: list) -> pd.DataFrame:

    engine = get_engine()
    all_dfs = []

    for ticker in tickers:
        # Load and engineer OHLCV features
        ohlcv = pd.read_sql(
            f"SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = '{ticker}' ORDER BY date",
            engine
        )
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv.sort_values("date", inplace=True)
        ohlcv.set_index("date", inplace=True)

        ohlcv          = build_ohlcv_features(ohlcv)
        ohlcv["label"] = make_label(ohlcv["close"])
        ohlcv.dropna(inplace=True)
        ohlcv["ticker"] = ticker

        # Load TF-IDF vectors
        tfidf_df = pd.read_sql(
            f"SELECT date, tfidf_vector FROM features WHERE ticker = '{ticker}' ORDER BY date",
            engine
        )
        tfidf_df["date"] = pd.to_datetime(tfidf_df["date"])
        tfidf_df.set_index("date", inplace=True)

        # Expand TF-IDF JSON into columns
        tfidf_expanded = pd.DataFrame(
            tfidf_df["tfidf_vector"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x or [])
            ).tolist(),
            index=tfidf_df.index
        ).fillna(0)
        tfidf_expanded.columns = [f"tfidf_{i}" for i in tfidf_expanded.columns]

    
        combined = ohlcv.join(tfidf_expanded, how="left")
        all_dfs.append(combined)

    return pd.concat(all_dfs)


def train_model():
    df = load_feature_matrix(TICKERS)

    TFIDF_COLS   = [c for c in df.columns if c.startswith("tfidf_")]
    ALL_FEATURES = OHLCV_FEATURE_COLS + TFIDF_COLS

    X = df[ALL_FEATURES].fillna(0).values
    y = df["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Dataset     : {X.shape[0]} samples")
    print(f"Features    : {len(OHLCV_FEATURE_COLS)} OHLCV + {len(TFIDF_COLS)} TF-IDF = {len(ALL_FEATURES)} total")

 
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    baseline_f1 = cross_val_score(baseline, X, y, cv=cv, scoring="f1").mean()
    print(f"\nBaseline F1 (default params): {baseline_f1:.4f}")

    # ── Stage 2: Tuned (GridSearchCV over key GBM hyperparameters) ───────────
    tuned_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth":     [3, 4, 5],
        "clf__subsample":     [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        tuned_pipeline,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)

    tuned_f1 = grid_search.best_score_
    print(f"Tuned F1    (grid-search best): {tuned_f1:.4f}")
    print(f"Improvement : +{tuned_f1 - baseline_f1:.4f}")
    print(f"Best params : {grid_search.best_params_}")

    joblib.dump({"model": grid_search.best_estimator_, "features": ALL_FEATURES}, "model.pkl")
    print("\nModel saved → model.pkl")

    return grid_search.best_estimator_, ALL_FEATURES


if __name__ == "__main__":
    train_model()