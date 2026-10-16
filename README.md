# Grocery Store Sales Forecasting and Demand Prediction Using Machine Learning

A beginner-friendly college project that uses Python, Pandas, Matplotlib, and Scikit-Learn to predict 7-day grocery demand and provide basic restocking recommendations.

---

## 📌 Project Overview
* **Domain:** Retail & Supply Chain Demand Forecasting
* **Dataset:** Small illustrative daily sales dataset with 5 basic items: **Rice, Milk, Bread, Sugar, and Eggs**
* **Machine Learning Algorithm:** `RandomForestRegressor` from Scikit-Learn
* **Visualization:** Matplotlib line plots showing historical sales and future 7-day predicted demand
* **Web Interface:** Clean, simple Streamlit web dashboard

> ⚠️ **Important Educational Note**:  
> The sample dataset (`grocery_sales.csv`) is illustrative. Predictions generated from this small dataset are demonstrations for learning machine learning concepts and are not real-world commercial forecasts.

---

## 📂 Project Structure
```text
supermarketsalesprediction/
├── grocery_sales.csv      # Small CSV dataset (Date, Item, Units_Sold, Price)
├── train_model.py         # Trains Random Forest and predicts next 7 days
├── app.py                 # Clean Streamlit dashboard
├── requirements.txt       # List of required Python libraries
├── grocery_model.joblib   # Saved trained Random Forest model
├── forecast_7days.csv     # Generated 7-day demand predictions
└── README.md              # Project guide and instructions
```

---

## 🚀 How to Run This Project

### 1. Install Required Libraries
Open your terminal (PowerShell or Command Prompt in VS Code / Antigravity) and run:
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
Run `train_model.py` to train the Random Forest regressor and generate the 7-day demand predictions:
```bash
python train_model.py
```
This script will:
* Load `grocery_sales.csv`
* Extract time features (`DayOfWeek`, `IsWeekend`, `Units_Sold_Yesterday`)
* Train the Random Forest Regressor
* Print the evaluation metrics (MAE and $R^2$)
* Save the trained model to `grocery_model.joblib`
* Save the 7-day forecast to `forecast_7days.csv`

### 3. Launch the Streamlit Dashboard
Run the following command to start the interactive web application:
```bash
streamlit run app.py
```
Your default web browser will open automatically at:
```
http://localhost:8501
```

---

## 💡 How Restocking Suggestions Work
In retail management, running out of stock leads to lost sales, while overstocking causes wastage.
The application applies a standard rule of thumb:
$$\text{Recommended Restock} = \text{Expected 7-Day Demand} + 20\% \text{ Safety Buffer}$$
* **Safety Buffer:** Protects against unexpected surges in customer traffic or vendor delivery delays.
* **Reorder Trigger:** If on-hand stock falls below expected 7-day demand, restock immediately.
