import yfinance as yf
import psycopg2
import feedparser
import pandas as pd
from datetime import date
from Config import DB_CONFIG, TICKERS, START_DATE, END_DATE


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def ingest_ohlcv(ticker: str):
    """Download OHLCV from Yahoo Finance and upsert into PostgreSQL."""
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.reset_index(inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    conn = get_connection()
    cur  = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO NOTHING
        """, (
            ticker,
            row["Date"],
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"])
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[OHLCV] Ingested {len(df)} rows for {ticker}")


def ingest_headlines(ticker: str):
    """
    Fetch headlines from Google News RSS for the ticker symbol.
    In a production system you would swap this for NewsAPI, Benzinga, etc.
    """
    url  = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    conn = get_connection()
    cur  = conn.cursor()

    for entry in feed.entries:
        pub = entry.get("published_parsed")
        if pub is None:
            continue
        pub_date = date(pub.tm_year, pub.tm_mon, pub.tm_mday)
        headline = entry.get("title", "")

        cur.execute("""
            INSERT INTO headlines (ticker, date, headline)
            VALUES (%s, %s, %s)
        """, (ticker, pub_date, headline))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[Headlines] Ingested {len(feed.entries)} headlines for {ticker}")


def run_ingestion():
    for ticker in TICKERS:
        ingest_ohlcv(ticker)
        ingest_headlines(ticker)


if __name__ == "__main__":
    run_ingestion()