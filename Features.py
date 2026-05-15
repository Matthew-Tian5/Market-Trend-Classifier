import psycopg2
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG, TICKERS
 
 
def get_connection():
    return psycopg2.connect(**DB_CONFIG)