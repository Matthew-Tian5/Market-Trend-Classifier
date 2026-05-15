import psycopg2
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import DB_CONFIG, TICKERS
from features import OHLCV_FEATURE_COLS

def get_connection():
    return psycopg2.connect(**DB_CONFIG)
 
 
def load_feature_matrix(tickers: list) -> pd.DataFrame:
    """
    Pulls all rows from the features table for the given tickers.
    Expands the TF-IDF JSONB column into individual float columns (tfidf_0, tfidf_1, ...).
    """
    conn = get_connection()
    rows = []
 
    for ticker in tickers:
        df = pd.read_sql(
            "SELECT * FROM features WHERE ticker = %s ORDER BY date",
            conn, params=(ticker,)
        )
        rows.append(df)
 
    conn.close()
    df = pd.concat(rows, ignore_index=True)
 
    tfidf_expanded = pd.DataFrame(
        df["tfidf_vector"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x or [])
        ).tolist()
    ).fillna(0)
    tfidf_expanded.columns = [f"tfidf_{i}" for i in tfidf_expanded.columns]
 
    return pd.concat([df.drop(columns=["tfidf_vector"]), tfidf_expanded], axis=1)








    def train_model():
        df = load_feature_matrix(TICKERS)
    
        TFIDF_COLS   = [c for c in df.columns if c.startswith("tfidf_")]
        ALL_FEATURES = OHLCV_FEATURE_COLS + TFIDF_COLS
    
        X = df[ALL_FEATURES].fillna(0).values
        y = df["label"].values
    
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
        print(f"Dataset     : {X.shape[0]} samples")
        print(f"Features    : {len(OHLCV_FEATURE_COLS)} OHLCV + {len(TFIDF_COLS)} TF-IDF = {len(ALL_FEATURES)} total")
    
        # ── Stage 1: Baseline (default GBM, no tuning) ───────────────────────────
        baseline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ])
        baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring="f1")
        baseline_f1     = baseline_scores.mean()
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
    
        # Save model and feature list for evaluation / inference
        joblib.dump({"model": grid_search.best_estimator_, "features": ALL_FEATURES}, "model.pkl")
        print("\nModel saved → model.pkl")
    
        return grid_search.best_estimator_, ALL_FEATURES
    
    
    if __name__ == "__main__":
        train_model()