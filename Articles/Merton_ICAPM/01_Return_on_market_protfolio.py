import yfinance as yf
import numpy as np

# Define the ticker symbol for the CAC40 index
ticker = "^FCHI"

# Download historical data (auto_adjust=True by default)
data = yf.download(ticker, start="2010-01-01", end="2025-09-22")

# Use the 'Close' column (already adjusted if auto_adjust=True)
data['Daily Return'] = data['Close'].pct_change()

# Drop the first NaN
data = data.dropna()

# Calculate the average daily return
average_daily_return = data['Daily Return'].mean()

# Annualize the daily return (252 trading days)
expected_annual_return = average_daily_return * 252

print(f"Expected Annual Return for CAC40: {expected_annual_return*100:.2f}%")
