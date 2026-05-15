import yfinance as yf
import psycopg2
import feedparser
import pandas as pd
from datetime import date
from config import DB_CONFIG, TICKERS, START_DATE, END_DATE
 
 
def get_connection():
    return psycopg2.connect(**DB_CONFIG)