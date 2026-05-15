import psycopg2
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG, TICKERS
 
 
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — momentum oscillator (0–100)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))