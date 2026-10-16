import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 90 days of daily sales for 5 grocery items (July 1, 2026 to September 28, 2026)
dates = pd.date_range(start='2026-07-01', periods=90, freq='D')

items_config = {
    'Milk': {'price': 2.50, 'base': 42, 'weekend_lift': 14, 'noise': 5},
    'Bread': {'price': 1.80, 'base': 36, 'weekend_lift': 10, 'noise': 4},
    'Eggs': {'price': 3.20, 'base': 28, 'weekend_lift': 8, 'noise': 4},
    'Rice': {'price': 12.00, 'base': 15, 'weekend_lift': 4, 'noise': 3},
    'Sugar': {'price': 4.00, 'base': 20, 'weekend_lift': 5, 'noise': 3}
}

rows = []
for d in dates:
    is_weekend = d.weekday() >= 5  # Saturday or Sunday
    for item, cfg in items_config.items():
        weekend_boost = cfg['weekend_lift'] if is_weekend else 0
        units = int(np.maximum(1, np.random.normal(cfg['base'] + weekend_boost, cfg['noise'])))
        rows.append({
            'Date': d.strftime('%Y-%m-%d'),
            'Item': item,
            'Units_Sold': units,
            'Price': cfg['price']
        })

df = pd.DataFrame(rows)
df.to_csv('grocery_sales.csv', index=False)
print(f"Created grocery_sales.csv with {len(df)} rows across {df['Item'].nunique()} items.")
