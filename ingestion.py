import yfinance as yf
import psycopg2
import feedparser
import pandas as pd
from datetime import date
from config import DB_CONFIG, TICKERS, START_DATE, END_DATE
 
 
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def ingest_ohlcv(ticker: str):

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.reset_index(inplace=True)
 
    conn = get_connection()
    cur  = conn.cursor()
 
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO NOTHING
        """, (
            ticker,
            row["Date"].date(),
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