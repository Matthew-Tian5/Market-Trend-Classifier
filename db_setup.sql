CREATE TABLE IF NOT EXISTS ohlcv (
    id      SERIAL PRIMARY KEY,
    ticker  VARCHAR(10),
    date    DATE,
    open    FLOAT,
    high    FLOAT,
    low     FLOAT,
    close   FLOAT,
    volume  BIGINT,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS headlines (
    id       SERIAL PRIMARY KEY,
    ticker   VARCHAR(10),
    date     DATE,
    headline TEXT
);

CREATE TABLE IF NOT EXISTS features (
    id            SERIAL PRIMARY KEY,
    ticker        VARCHAR(10),
    date          DATE,
    rsi           FLOAT,
    macd          FLOAT,
    macd_signal   FLOAT,
    rolling_vol   FLOAT,
    tfidf_vector  JSONB,
    label         INT,
    UNIQUE(ticker, date)
);