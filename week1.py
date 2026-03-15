import yfinance as yf
import numpy as np
import pandas as pd
import time

TICKERS = ["AAPL", "JNJ", "XOM", "SPY", "BTC-USD"]
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"

def fetch_data(tickers, start, end, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                threads=True
            )
            return data
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise Exception("Data download failed after retries.")

raw_data = fetch_data(TICKERS, START_DATE, END_DATE)

price_df = pd.DataFrame()

for ticker in TICKERS:
    if ticker == "BTC-USD":
        price_df[ticker] = raw_data[ticker]["Close"]
    else:
        price_df[ticker] = raw_data[ticker]["Close"]

price_df = price_df.ffill()

price_df = price_df.dropna()

price_matrix = price_df.values

log_returns = np.log(price_matrix[1:] / price_matrix[:-1])

mean_returns = np.mean(log_returns, axis=0)

cov_matrix = np.cov(log_returns.T)

weights = np.ones(len(TICKERS)) / len(TICKERS)

portfolio_return = np.dot(mean_returns, weights)

portfolio_volatility = np.sqrt(
    np.dot(weights.T, np.dot(cov_matrix, weights))
)

print("Expected Daily Return:", portfolio_return)
print("Daily Volatility:", portfolio_volatility)

price_df.to_csv("cleaned_price_data.csv")
pd.DataFrame(log_returns, columns=TICKERS).to_csv("log_returns.csv")

price_df = price_df.ffill()

price_df = price_df.bfill()

z_scores = np.abs((log_returns - log_returns.mean()) / log_returns.std())
log_returns[z_scores > 5] = 0
