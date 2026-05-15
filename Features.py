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



def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """
    MACD = EMA(fast) - EMA(slow).
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast  = close.ewm(span=fast, adjust=False).mean()
    ema_slow  = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - sig_line
    return macd_line, sig_line, histogram



def compute_bollinger_bands(close: pd.Series, window: int = 20):
    """
    Bollinger Bands — normalized band width and %B position.
    bb_width: how wide the bands are relative to the SMA.
    bb_pct:   where the price sits within the bands (0 = lower, 1 = upper).
    """
    sma   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = (upper - lower) / sma
    bb_pct   = (close - lower) / (upper - lower)
    return bb_width, bb_pct
 

def compute_rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualized rolling standard deviation of log returns."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)
 
 
def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume in the direction of price movement."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()
 
 