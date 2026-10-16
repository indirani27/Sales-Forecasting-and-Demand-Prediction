"""
Grocery Store Sales Forecasting and Demand Prediction
Machine Learning Pipeline with Intraday Hourly Demand Flow
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
print("=" * 75)
print("  GROCERY STORE SALES FORECASTING & DEMAND PREDICTION SYSTEM")
print("=" * 75)

df = pd.read_csv("grocery_sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

items = ['Milk', 'Bread', 'Eggs', 'Rice', 'Sugar']
print(f"Dataset: {len(df)} records across 5 grocery items ({', '.join(items)})")
print(f"Historical Horizon: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# ---------------------------------------------------------------------------
# 2. Feature Engineering & Daily Model Training
# ---------------------------------------------------------------------------
df['DayOfWeek'] = df['Date'].dt.weekday
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['DayOfMonth'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

item_mapping = {'Milk': 0, 'Bread': 1, 'Eggs': 2, 'Rice': 3, 'Sugar': 4}
df['Item_Code'] = df['Item'].map(item_mapping)

# Compute previous day's sales (Lag 1)
df = df.sort_values(by=['Item', 'Date']).reset_index(drop=True)
df['Units_Sold_Yesterday'] = df.groupby('Item')['Units_Sold'].shift(1)
item_means = df.groupby('Item')['Units_Sold'].transform('mean')
df['Units_Sold_Yesterday'] = df['Units_Sold_Yesterday'].fillna(item_means)
df = df.sort_values(by=['Date', 'Item']).reset_index(drop=True)

feature_cols = ['Item_Code', 'DayOfWeek', 'IsWeekend', 'Price', 'Units_Sold_Yesterday']
X = df[feature_cols]
y = df['Units_Sold']

# Chronological 80/20 train-test split
unique_dates = df['Date'].drop_duplicates().sort_values()
split_date = unique_dates.iloc[int(len(unique_dates) * 0.8)]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X[train_mask], y[train_mask])

test_preds = model.predict(X[test_mask])
mae = mean_absolute_error(y[test_mask], test_preds)
r2 = r2_score(y[test_mask], test_preds)

print(f"\nModel Performance:")
print(f"  * Algorithm:             Random Forest Regressor")
print(f"  * Mean Absolute Error:   {mae:.2f} units/day")
print(f"  * Accuracy (R2 Score):   {r2:.3f} ({r2*100:.1f}%)")

# Train final production model on all historical data
final_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
final_model.fit(X, y)
joblib.dump(final_model, "grocery_model.joblib")

# ---------------------------------------------------------------------------
# 3. Forecast Next 7 Days Demand
# ---------------------------------------------------------------------------
latest_date = df['Date'].max()
future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=7, freq='D')

forecast_records = []
for item_name in items:
    item_code = item_mapping[item_name]
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

        forecast_records.append({
            'Date': f_date.strftime('%Y-%m-%d'),
            'Day': f_date.strftime('%A'),
            'Item': item_name,
            'Predicted_Units': pred,
            'Unit_Price': price,
            'Revenue': round(pred * price, 2),
            'IsWeekend': is_wknd
        })
        last_units = pred

forecast_df = pd.DataFrame(forecast_records)
forecast_df.to_csv("forecast_7days.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Display 7-Day Predicted Demand
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  PREDICTED DEMAND FOR THE NEXT 7 DAYS (UNITS)")
print("=" * 75)

pivot_forecast = forecast_df.pivot(index='Item', columns='Date', values='Predicted_Units')
print(pivot_forecast.to_string())

# ---------------------------------------------------------------------------
# 5. Intraday Hourly Demand Flow Mathematical Profiles
# ---------------------------------------------------------------------------
# Store operating hours: 6:00 AM (6) to 10:00 PM (22)
hours_smooth = np.linspace(6, 22, 100)

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Hourly distribution profile generator:
# - Milk: High morning peak (6-9 AM), high evening peak (5-8 PM), quiet in afternoon
# - Bread: High morning peak (7-9:30 AM), moderate afternoon/evening
# - Eggs: Morning breakfast peak, moderate evening cooking peak
# - Rice: Steady low frequency, slight midday & early evening grocery pickups
# - Sugar: Steady low frequency throughout the day
def get_intraday_curve(item_name, daily_total):
    if item_name == 'Milk':
        # Bimodal peak: 7:45 AM and 6:15 PM
        flow = gaussian(hours_smooth, 7.8, 1.1, 8.2) + gaussian(hours_smooth, 18.2, 1.2, 7.2) + 0.9
    elif item_name == 'Bread':
        # Morning peak around 8:15 AM
        flow = gaussian(hours_smooth, 8.2, 1.3, 6.0) + gaussian(hours_smooth, 17.5, 1.8, 2.5) + 1.2
    elif item_name == 'Eggs':
        # Morning peak around 8:00 AM, smaller evening peak around 6:00 PM
        flow = gaussian(hours_smooth, 8.0, 1.2, 4.8) + gaussian(hours_smooth, 18.0, 1.4, 2.8) + 0.9
    elif item_name == 'Rice':
        # Low frequency staple
        flow = gaussian(hours_smooth, 12.0, 2.2, 1.4) + gaussian(hours_smooth, 18.0, 1.8, 1.3) + 0.5
    else:  # Sugar
        # Low, flat frequency
        flow = gaussian(hours_smooth, 13.0, 3.5, 1.1) + 0.7

    # Scale the curve so area equals daily forecast
    area = np.trapezoid(flow, hours_smooth)
    scaled_flow = (flow / area) * daily_total
    return scaled_flow

print("\n" + "=" * 75)
print("  INTRADAY DEMAND FLOW & PEAK SALES WINDOWS")
print("=" * 75)

flow_summary = []
for item_name in items:
    daily_avg = forecast_df[forecast_df['Item'] == item_name]['Predicted_Units'].mean()
    curve = get_intraday_curve(item_name, daily_avg)

    # Compute key time-of-day volumes
    morning_mask = (hours_smooth >= 6) & (hours_smooth <= 9)
    afternoon_mask = (hours_smooth > 9) & (hours_smooth < 17)
    evening_mask = (hours_smooth >= 17) & (hours_smooth <= 20)

    morning_vol = np.trapezoid(curve[morning_mask], hours_smooth[morning_mask])
    afternoon_vol = np.trapezoid(curve[afternoon_mask], hours_smooth[afternoon_mask])
    evening_vol = np.trapezoid(curve[evening_mask], hours_smooth[evening_mask])

    peak_idx = int(np.argmax(curve))
    peak_time_val = hours_smooth[peak_idx]
    peak_hr = int(peak_time_val)
    peak_min = int((peak_time_val - peak_hr) * 60)
    peak_str = f"{peak_hr if peak_hr <= 12 else peak_hr-12}:{peak_min:02d} {'AM' if peak_hr < 12 else 'PM'}"
    peak_rate = curve[peak_idx]

    flow_summary.append({
        'Item': item_name,
        'Daily Forecast': f"{daily_avg:.1f} units",
        'Morning (6-9 AM)': f"{(morning_vol/daily_avg)*100:.1f}%",
        'Afternoon (9-5 PM)': f"{(afternoon_vol/daily_avg)*100:.1f}%",
        'Evening (5-8 PM)': f"{(evening_vol/daily_avg)*100:.1f}%",
        'Peak Hour': f"{peak_str} ({peak_rate:.1f} u/hr)"
    })

flow_table = pd.DataFrame(flow_summary)
print(flow_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Recommended Restocking Plan
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  RECOMMENDED RESTOCKING & INVENTORY PLAN")
print("=" * 75)

restock_list = []
for item_name in items:
    sub = forecast_df[forecast_df['Item'] == item_name]
    total_demand = sub['Predicted_Units'].sum()
    buffer_units = int(np.ceil(total_demand * 0.20))
    restock_units = total_demand + buffer_units
    price = sub['Unit_Price'].iloc[0]
    total_cost = restock_units * price

    restock_list.append({
        'Item': item_name,
        'Price': f"${price:.2f}",
        '7-Day Demand': f"{total_demand} units",
        'Safety Buffer': f"+{buffer_units} units",
        'Recommended Stock': f"{restock_units} units",
        'Stock Value': f"${total_cost:,.2f}"
    })

restock_table = pd.DataFrame(restock_list)
print(restock_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Generate Individual Demand Flow Graphs for Each of the 5 Items
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  GENERATING INDIVIDUAL DEMAND FLOW GRAPHS FOR ALL 5 ITEMS...")
print("=" * 75)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

item_colors = {
    'Milk': '#1f77b4',
    'Bread': '#e67e22',
    'Eggs': '#27ae60',
    'Rice': '#8e44ad',
    'Sugar': '#d35400'
}

hour_ticks = [6, 8, 10, 12, 14, 16, 18, 20, 22]
hour_tick_labels = ['6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM']

individual_files = []

# Generate each graph individually with a single solid line
for item_name in items:
    daily_avg = forecast_df[forecast_df['Item'] == item_name]['Predicted_Units'].mean()
    curve = get_intraday_curve(item_name, daily_avg)
    color = item_colors[item_name]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.canvas.manager.set_window_title(f"{item_name} - Demand Flow Graph")

    # Single solid line describing the flow
    ax.plot(hours_smooth, curve, color=color, linewidth=2.8, label=f'{item_name} Demand Flow (Units/Hour)')
    ax.fill_between(hours_smooth, curve, color=color, alpha=0.15)

    # Detailed annotations describing flow
    if item_name == 'Milk':
        # High early morning peak
        p1_idx = int(np.argmax(curve[(hours_smooth >= 6) & (hours_smooth <= 10)]))
        p1_x = hours_smooth[(hours_smooth >= 6) & (hours_smooth <= 10)][p1_idx]
        p1_y = curve[(hours_smooth >= 6) & (hours_smooth <= 10)][p1_idx]
        ax.annotate('Morning Rush Peak\n(Early Morning 6-9 AM)',
                    xy=(p1_x, p1_y), xytext=(p1_x - 1.2, p1_y + 1.2),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.8),
                    fontsize=9.5, fontweight='bold', color='#c0392b')

        # High evening peak
        p2_idx = int(np.argmax(curve[(hours_smooth >= 16) & (hours_smooth <= 21)]))
        p2_x = hours_smooth[(hours_smooth >= 16) & (hours_smooth <= 21)][p2_idx]
        p2_y = curve[(hours_smooth >= 16) & (hours_smooth <= 21)][p2_idx]
        ax.annotate('Evening Rush Peak\n(Evening 5-8 PM)',
                    xy=(p2_x, p2_y), xytext=(p2_x - 2.0, p2_y + 1.2),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.8),
                    fontsize=9.5, fontweight='bold', color='#c0392b')

        # Low afternoon dip annotation
        ax.annotate('Low Afternoon Frequency (Quiet Period)', xy=(13.0, 1.2), xytext=(10.5, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2),
                    fontsize=9, color='#555555')

        ax.axvspan(6, 9, color='#e8f4f8', alpha=0.55, label='Morning Peak (Breakfast)')
        ax.axvspan(17, 20, color='#fef5e7', alpha=0.55, label='Evening Peak (Tea/Dinner)')

    elif item_name == 'Bread':
        ax.annotate('Morning Breakfast Peak (7-10 AM)', xy=(8.2, max(curve)), xytext=(8.5, max(curve) + 0.8),
                    arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.5), fontsize=9.5, fontweight='bold', color='#d35400')
        ax.axvspan(7, 10, color='#fef5e7', alpha=0.5, label='Breakfast Peak Window')

    elif item_name == 'Eggs':
        ax.annotate('Morning Peak (7-9 AM)', xy=(8.0, max(curve)), xytext=(8.5, max(curve) + 0.6),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5), fontsize=9.5, fontweight='bold', color='#27ae60')

    elif item_name in ['Rice', 'Sugar']:
        mid_idx = len(hours_smooth) // 2
        ax.annotate('Steady Low Frequency Demand\n(Pantry Staples Purchased Periodically)',
                    xy=(hours_smooth[mid_idx], curve[mid_idx]), xytext=(hours_smooth[mid_idx] - 3.5, curve[mid_idx] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5), fontsize=9.5, fontweight='bold', color='#2c3e50')

    ax.set_title(f"{item_name.upper()} - Intraday Demand Flow Curve\nDaily Forecast: {daily_avg:.1f} Units/Day | Single Solid Line Velocity",
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Time of Day (Operating Hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Sales Velocity (Units Sold per Hour)", fontsize=10, fontweight='bold')
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_tick_labels, fontsize=9)
    ax.set_xlim(6, 22)
    ax.set_ylim(0, max(curve) * 1.35)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    chart_path = f"{item_name.lower()}_demand_flow.png"
    plt.savefig(chart_path, dpi=300)
    individual_files.append(chart_path)
    print(f"-> Saved individual graph: {chart_path}")

# ---------------------------------------------------------------------------
# 8. Generate Comparative Intraday Graph (All 5 Items on One Canvas)
# ---------------------------------------------------------------------------
fig_all, ax_all = plt.subplots(figsize=(12, 6))
fig_all.canvas.manager.set_window_title("Comparative Intraday Demand Flow - All 5 Grocery Items")

for item_name in items:
    daily_avg = forecast_df[forecast_df['Item'] == item_name]['Predicted_Units'].mean()
    curve = get_intraday_curve(item_name, daily_avg)
    lw = 3.2 if item_name == 'Milk' else 1.8
    ax_all.plot(hours_smooth, curve, label=f"{item_name} ({daily_avg:.1f} u/day)", color=item_colors[item_name], linewidth=lw)

# Highlight Milk's high early morning and evening peaks
ax_all.axvspan(6, 9, color='#1f77b4', alpha=0.08, label='Early Morning Rush (6-9 AM)')
ax_all.axvspan(17, 20, color='#e67e22', alpha=0.08, label='Evening Rush (5-8 PM)')

ax_all.set_title("Comparative Demand Flow: Milk Surges in Early Morning & Evening\n(In the mean time, Rice, Sugar, Bread, and Eggs sell at lower frequency)",
                 fontsize=12, fontweight='bold', pad=12)
ax_all.set_xlabel("Time of Day (6:00 AM to 10:00 PM)", fontsize=10, fontweight='bold')
ax_all.set_ylabel("Sales Velocity (Units Sold per Hour)", fontsize=10, fontweight='bold')
ax_all.set_xticks(hour_ticks)
ax_all.set_xticklabels(hour_tick_labels, fontsize=9)
ax_all.set_xlim(6, 22)
ax_all.grid(True, linestyle=':', alpha=0.6)
ax_all.legend(loc='upper right', fontsize=9)

plt.tight_layout()
comp_file = "grocery_demand_flow_comparison.png"
plt.savefig(comp_file, dpi=300)
print(f"-> Saved comparison graph: {comp_file}")

# ---------------------------------------------------------------------------
# 9. Display Graphs Immediately on Screen
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  DISPLAYING DEMAND FLOW GRAPHS IMMEDIATELY...")
print("=" * 75)

try:
    if hasattr(os, 'startfile'):
        # Open comparison and milk individual chart directly
        os.startfile(comp_file)
        os.startfile("milk_demand_flow.png")
    plt.show()
except Exception:
    if hasattr(os, 'startfile'):
        os.startfile(comp_file)

print("-> Individual demand flow graphs and comparison chart displayed successfully.")
print("=" * 75 + "\n")
