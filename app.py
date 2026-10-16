"""
=============================================================================
Project: Grocery Store Sales Forecasting and Demand Prediction
Script: app.py
Description: Simple, clean Streamlit dashboard for visualizing grocery sales,
             reviewing 7-day predicted demand, and calculating restocking needs.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grocery Sales Forecasting & Demand Prediction",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Header & Project Information
# ---------------------------------------------------------------------------
st.title("🛒 Grocery Store Sales Forecasting and Demand Prediction")
st.write("A beginner-friendly machine learning project using Python, Pandas, and Random Forest.")

# Beginner / Student Educational Notice
st.info(
    "📌 **Educational Note**: This project uses a small, illustrative dataset of 5 basic grocery items "
    "(Rice, Milk, Bread, Sugar, and Eggs). Predictions generated here demonstrate machine learning "
    "concepts and supply chain principles. They are educational demonstrations, not real-world forecasts."
)

# ---------------------------------------------------------------------------
# Load Dataset & Predictions
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load historical sales data
    sales_df = pd.read_csv("grocery_sales.csv")
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])

    # Load pre-computed 7-day forecast
    forecast_df = pd.read_csv("forecast_7days.csv")
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    return sales_df, forecast_df

sales_df, forecast_df = load_data()

# ---------------------------------------------------------------------------
# Section 1: Dataset Overview
# ---------------------------------------------------------------------------
st.header("1. Dataset Overview")
st.write("Preview of the daily grocery sales records stored in `grocery_sales.csv`:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(sales_df)} rows")
with col2:
    st.metric("Number of Items", f"{sales_df['Item'].nunique()} items")
with col3:
    st.metric("Date Range", f"{sales_df['Date'].min().strftime('%b %d')} to {sales_df['Date'].max().strftime('%b %d, %Y')}")
with col4:
    total_units = sales_df['Units_Sold'].sum()
    st.metric("Total Historical Units Sold", f"{total_units:,} units")

# Expandable raw dataset preview
with st.expander("🔍 Click to view first 10 rows of raw dataset"):
    st.dataframe(sales_df.head(10), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Item Selection
# ---------------------------------------------------------------------------
st.header("2. Select a Grocery Item")
items_list = sorted(sales_df['Item'].unique())
selected_item = st.selectbox("Choose an item to analyze:", items_list, index=items_list.index("Milk") if "Milk" in items_list else 0)

# Filter historical and forecast data for the selected item
item_history = sales_df[sales_df['Item'] == selected_item].sort_values(by='Date')
item_forecast = forecast_df[forecast_df['Item'] == selected_item].sort_values(by='Date')
item_price = item_history['Price'].iloc[-1]

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Sales Graph using Matplotlib
# ---------------------------------------------------------------------------
st.header(f"3. Historical Sales & 7-Day Predicted Demand for {selected_item}")
st.write("The graph below plots daily historical sales (blue) and the Random Forest model's next 7 days forecast (red dashed line).")

# Matplotlib figure creation
fig, ax = plt.subplots(figsize=(11, 4.8))

# 1. Plot historical sales
ax.plot(
    item_history['Date'],
    item_history['Units_Sold'],
    color='#1f77b4',
    linewidth=2,
    label='Historical Daily Sales'
)

# 2. Connect the last historical point with the forecast
connect_dates = [item_history['Date'].iloc[-1]] + list(item_forecast['Date'])
connect_values = [item_history['Units_Sold'].iloc[-1]] + list(item_forecast['Predicted_Units'])

ax.plot(
    connect_dates,
    connect_values,
    color='#e74c3c',
    linewidth=2.5,
    linestyle='--',
    marker='o',
    markersize=6,
    label='Random Forest 7-Day Forecast'
)

# Formatting chart
ax.set_title(f"Sales & Demand Forecast: {selected_item} (Unit Price: ${item_price:.2f})", fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Units Sold / Demand", fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', frameon=True)
fig.autofmt_xdate()
plt.tight_layout()

# Display Matplotlib plot in Streamlit
st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: 7-Day Predicted Demand Table & Metrics
# ---------------------------------------------------------------------------
st.header(f"4. Next 7 Days Demand Breakdown ({selected_item})")

total_predicted = int(item_forecast['Predicted_Units'].sum())
avg_daily_predicted = float(item_forecast['Predicted_Units'].mean())
estimated_revenue = float(total_predicted * item_price)

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Total Expected Demand (Next 7 Days)", f"{total_predicted} units")
with m_col2:
    st.metric("Average Expected Daily Demand", f"{avg_daily_predicted:.1f} units / day")
with m_col3:
    st.metric("Estimated 7-Day Revenue", f"${estimated_revenue:,.2f}")

# Clean display table
forecast_display = item_forecast[['Date', 'Predicted_Units']].copy()
forecast_display['Day'] = forecast_display['Date'].dt.strftime('%A')
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
forecast_display['Expected Revenue ($)'] = (forecast_display['Predicted_Units'] * item_price).map("${:,.2f}".format)
forecast_display = forecast_display[['Date', 'Day', 'Predicted_Units', 'Expected Revenue ($)']]

st.dataframe(forecast_display, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5: Basic Restocking Suggestion
# ---------------------------------------------------------------------------
st.header("5. Basic Restocking Suggestion")
st.write(
    "To prevent stockouts (running out of goods) while avoiding overstocking (waste or spoilage), "
    "retail managers maintain a **safety buffer** above expected demand."
)

# Simple Restocking Formula:
# Suggested Order = Expected 7-Day Demand + 20% Safety Buffer
safety_buffer = int(np.ceil(total_predicted * 0.20))
suggested_restock = total_predicted + safety_buffer

r_col1, r_col2 = st.columns([1, 1])

with r_col1:
    st.success(f"""
    ### 📦 Restocking Recommendation for **{selected_item}**:
    * **Expected 7-Day Demand:** **{total_predicted} units**
    * **Recommended Safety Buffer (+20%):** **{safety_buffer} units**
    * **Suggested Stock to Maintain:** **{suggested_restock} units**
    """)

with r_col2:
    st.info(f"""
    ### 💡 Supply Chain Rule of Thumb:
    * If current on-hand stock is **below {total_predicted} units**, place a purchase order immediately.
    * If current on-hand stock is **between {total_predicted} and {suggested_restock} units**, stock is adequate.
    * If current stock exceeds **{suggested_restock} units**, pause reordering to avoid excess holding.
    """)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6: How the Model Works (Beginner Guide)
# ---------------------------------------------------------------------------
st.header("6. How This Model Works (Beginner Explanation)")
st.write("""
1. **Inputs (Features):**
   * **Item Code:** Tells the model which product is being predicted.
   * **Day of the Week:** Learns that demand usually increases on weekends (Friday, Saturday, Sunday).
   * **Previous Day's Sales:** Uses yesterday's demand to understand current sales momentum.
   * **Price:** Relates product pricing to consumer purchasing behavior.

2. **Algorithm (Random Forest Regressor):**
   * Combines 100 individual decision trees that vote together to produce a stable, reliable numerical prediction.

3. **Evaluation Metric:**
   * **MAE (Mean Absolute Error):** Measures the average difference between the model's prediction and actual units sold.
""")

st.caption("College Project: Grocery Store Sales Forecasting and Demand Prediction Using Machine Learning")
