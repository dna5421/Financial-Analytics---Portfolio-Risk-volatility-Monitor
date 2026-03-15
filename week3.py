import pandas as pd

data = {
    "Sector": ["Technology", "Healthcare", "Energy"],
    "Investment": [50000, 30000, 20000]
}

df = pd.DataFrame(data)

# Save output
df.to_csv("portfolio_data.csv", index=False)