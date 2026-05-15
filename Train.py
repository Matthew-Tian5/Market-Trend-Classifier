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