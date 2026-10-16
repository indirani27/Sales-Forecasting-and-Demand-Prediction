"""
Grocery Store Sales Forecasting and Demand Prediction
Script: train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("grocery_sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
df['DayOfWeek'] = df['Date'].dt.weekday
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['DayOfMonth'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

item_mapping = {'Rice': 0, 'Milk': 1, 'Bread': 2, 'Sugar': 3, 'Eggs': 4}
df['Item_Code'] = df['Item'].map(item_mapping)

df = df.sort_values(by=['Item', 'Date']).reset_index(drop=True)
df['Units_Sold_Yesterday'] = df.groupby('Item')['Units_Sold'].shift(1)
item_means = df.groupby('Item')['Units_Sold'].transform('mean')
df['Units_Sold_Yesterday'] = df['Units_Sold_Yesterday'].fillna(item_means)
df = df.sort_values(by=['Date', 'Item']).reset_index(drop=True)

# Train-Test Split & Modeling
features = ['Item_Code', 'DayOfWeek', 'IsWeekend', 'Price', 'Units_Sold_Yesterday']
X = df[features]
y = df['Units_Sold']

unique_dates = df['Date'].drop_duplicates().sort_values()
split_date = unique_dates.iloc[int(len(unique_dates) * 0.8)]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X[train_mask], y[train_mask])

test_preds = model.predict(X[test_mask])
mae = mean_absolute_error(y[test_mask], test_preds)
r2 = r2_score(y[test_mask], test_preds)

print(f"Random Forest Regressor trained.")
print(f"MAE: {mae:.2f} units | R2 Score: {r2:.3f}")

# Final Model
final_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
final_model.fit(X, y)
joblib.dump(final_model, "grocery_model.joblib")

# 7-Day Forecast
latest_date = df['Date'].max()
future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=7, freq='D')

forecast_rows = []
for item_name, item_code in item_mapping.items():
    item_df = df[df['Item'] == item_name].sort_values(by='Date')
    price = item_df['Price'].iloc[-1]
    last_units = item_df['Units_Sold'].iloc[-1]

    for f_date in future_dates:
        dow = f_date.weekday()
        is_wknd = 1 if dow >= 5 else 0

        feat = pd.DataFrame([{
            'Item_Code': item_code,
            'DayOfWeek': dow,
            'IsWeekend': is_wknd,
            'Price': price,
            'Units_Sold_Yesterday': last_units
        }])

        pred = int(np.round(final_model.predict(feat)[0]))
        pred = max(1, pred)

        forecast_rows.append({
            'Date': f_date.strftime('%Y-%m-%d'),
            'Item': item_name,
            'Predicted_Units': pred,
            'Price': price
        })
        last_units = pred

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("forecast_7days.csv", index=False)
print("7-day forecast saved to forecast_7days.csv.")
