import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('data.csv', parse_dates=['Date'])
except FileNotFoundError:
    dates = pd.date_range(start='2023-01-01', periods=300)
    prices = 100 + np.cumsum(np.random.randn(300))
    data = pd.DataFrame({'Date': dates, 'Close': prices})

data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

trading_days = 252
mu = data['Log_Return'].mean() 
sigma = data['Log_Return'].std() 

S0 = data['Close'].iloc[-1]
T = 1.0 
N = trading_days
dt = T/N
simulations = 10000

price_paths = np.zeros((N + 1, simulations))
price_paths[0] = S0

for t in range(1, N + 1):
    Z = np.random.standard_normal(simulations)
    price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * Z))

final_prices = price_paths[-1]
returns = (final_prices - S0) / S0
VaR_95 = np.percentile(returns, 5)

print(f"Annualized Volatility: {sigma * np.sqrt(252):.2%}")
print(f"Value at Risk (95%): {VaR_95:.2%}")

plt.figure(figsize=(10,5))
plt.plot(price_paths[:, :100])
plt.axhline(S0, color='black', linestyle='--', label='Start Price')
plt.title(f"Monte Carlo Simulation ({simulations} paths)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()