import psycopg2
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import DB_CONFIG, TICKERS


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    c = DB_CONFIG
    url = f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
    return create_engine(url)




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
    sma      = close.rolling(window).mean()
    std      = close.rolling(window).std()
    upper    = sma + 2 * std
    lower    = sma - 2 * std
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


def build_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all 19 engineered features to a raw OHLCV dataframe.
    Input must have columns: open, high, low, close, volume.
    """
    close  = df["close"]
    volume = df["volume"]
    high   = df["high"]
    low    = df["low"]

    # Momentum
    df["rsi_14"] = compute_rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    # Trend / moving averages
    df["sma_10"]         = close.rolling(10).mean()
    df["sma_50"]         = close.rolling(50).mean()
    df["ema_12"]         = close.ewm(span=12, adjust=False).mean()
    df["price_vs_sma10"] = close / df["sma_10"]
    df["price_vs_sma50"] = close / df["sma_50"]

    # Volatility
    df["rolling_vol_5"]          = compute_rolling_volatility(close, 5)
    df["rolling_vol_20"]         = compute_rolling_volatility(close, 20)
    df["bb_width"], df["bb_pct"] = compute_bollinger_bands(close)

    # Volume
    df["volume_change"] = volume.pct_change()
    df["volume_ratio"]  = volume / volume.rolling(20).mean()
    df["obv"]           = compute_obv(close, volume)

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)

    # Intraday range
    df["hl_range"] = (high - low) / close

    return df


def make_label(close: pd.Series) -> pd.Series:
    """Binary label: 1 if tomorrow's close > today's close, else 0."""
    return (close.shift(-1) > close).astype(int)



OHLCV_FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "sma_10", "sma_50", "ema_12", "price_vs_sma10", "price_vs_sma50",
    "rolling_vol_5", "rolling_vol_20", "bb_width", "bb_pct",
    "volume_change", "volume_ratio", "obv",
    "return_1d", "return_5d", "hl_range"
]



def build_tfidf_per_date(ticker: str) -> dict:
    """
    Groups all headlines for a ticker by date, fits a TF-IDF vectorizer,
    and returns {date: vector_as_list}.
    """
    engine = get_engine()
    df     = pd.read_sql(
        f"SELECT date, headline FROM headlines WHERE ticker = '{ticker}' ORDER BY date",
        engine
    )

    if df.empty:
        return {}

    grouped    = df.groupby("date")["headline"].apply(lambda x: " ".join(x)).reset_index()
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_mat  = vectorizer.fit_transform(grouped["headline"])

    return {
        row["date"]: tfidf_mat[i].toarray().flatten().tolist()
        for i, row in grouped.iterrows()
    }



def build_and_store_features(ticker: str):
    engine = get_engine()
    df     = pd.read_sql(
        f"SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = '{ticker}' ORDER BY date",
        engine
    )

    if df.empty:
        print(f"[Features] No OHLCV data found for {ticker}")
        return

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    df          = build_ohlcv_features(df)
    df["label"] = make_label(df["close"])
    df.dropna(inplace=True)

    tfidf_map = build_tfidf_per_date(ticker)

    conn = get_connection()
    cur  = conn.cursor()

    for idx_date, row in df.iterrows():
        date_key  = idx_date.date()
        tfidf_vec = tfidf_map.get(date_key, [])

        cur.execute("""
            INSERT INTO features (ticker, date, rsi, macd, macd_signal, rolling_vol, tfidf_vector, label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE
                SET rsi=EXCLUDED.rsi, macd=EXCLUDED.macd,
                    macd_signal=EXCLUDED.macd_signal,
                    rolling_vol=EXCLUDED.rolling_vol,
                    tfidf_vector=EXCLUDED.tfidf_vector,
                    label=EXCLUDED.label
        """, (
            ticker, date_key,
            float(row["rsi_14"]),
            float(row["macd"]),
            float(row["macd_signal"]),
            float(row["rolling_vol_20"]),
            json.dumps(tfidf_vec),
            int(row["label"])
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[Features] {ticker} — {len(df)} rows, {len(OHLCV_FEATURE_COLS)} OHLCV features")


def run_feature_engineering():
    for ticker in TICKERS:
        build_and_store_features(ticker)


if __name__ == "__main__":
    run_feature_engineering()