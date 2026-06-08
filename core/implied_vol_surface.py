import yfinance as yf, pandas as pd, matplotlib.pyplot as plt
from black_scholes import implied_vol
from datetime import datetime

# Fetch the stock data ∫
ticker = yf.Ticker("SPY")
#print(ticker.options)

# Pick an expiry date
expiry = ticker.options[12] # pick the third date in the list,something that is 30-60 days away, that is farther from expiry.
#print(expiry)

# Fetch the option chain: getting all the call and put contracts for this expiry date: every available strike price in the market 
chain = ticker.option_chain(expiry)
calls = chain.calls 
puts = chain.puts
#print(calls.columns.tolist())

# Get the current stock price S_0: need it as input to the implied_vol() function 
spot = ticker.history(period = "1d")["Close"].iloc[-1]
print(spot)

# Calculate the time to expiry T by converting the expiry date string into T in years
today = datetime.today() # get today's date
expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
T = (expiry_date - today).days / 365 #days extracts the number of days between the 2 dates 
print(T)

# Obtain the risk free rate: need the input r as an input to the implied_vol: for now, no dynamically fetching from rates API, just hardcode
r = 0.045 # 4.5%

# Compute implied volatility for each strike 
results = []

for _, row in calls.iterrows():
    strike = row['strike']
    mid = row['lastPrice'] if (row['bid'] == 0 and row['ask'] == 0) else (row['bid'] + row['ask']) / 2
    
    if mid > 0:  # skip options with no market price
        iv = implied_vol(mid, spot, strike, T, r, option_type='call')
        results.append({'strike': strike, 'iv': iv})

iv_df = pd.DataFrame(results)
print(f"Strike: {strike}, Bid: {row['bid']}, Ask: {row['ask']}, Mid: {mid}")
print(iv_df)

# Plot the vol surface 
plt.figure(figsize=(12, 6))
plt.plot(iv_df['strike'], iv_df['iv'] * 100)
plt.axvline(x=spot, color='red', linestyle='--', label=f'Spot: ${spot:.0f}')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility (%)')
plt.title(f'SPY Implied Volatility Surface — Expiry {expiry}')
plt.legend()
plt.grid(True)
plt.show()

filtered = iv_df[(iv_df['strike'] > spot * 0.80) & (iv_df['strike'] < spot * 1.20)]
plt.plot(filtered['strike'], filtered['iv'] * 100)