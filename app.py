import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT/"data/walmart_sales.csv", parse_dates=["Date"])
model_pack = joblib.load(ROOT/"models/random_forest.joblib")
model = model_pack["model"]
features = model_pack["features"]
summary = json.loads((ROOT/"outputs/summary.json").read_text())
comparison = pd.read_csv(ROOT/"outputs/model_comparison.csv")
importance = pd.read_csv(ROOT/"outputs/feature_importance.csv")
pred = pd.read_csv(ROOT/"outputs/predictions.csv", parse_dates=["Date"])

st.title("📈 AI-Driven Retail Sales Forecasting")
st.caption("End-to-end Machine Learning Web Application | Python • Pandas • NumPy • Scikit-learn")

tabs = st.tabs(["🏠 Overview","📊 Dataset","🧹 Preprocessing","🧩 Features","🤖 Model","📈 Output","🔮 Predict"])

with tabs[0]:
    st.subheader("Project Objective")
    st.write("Predict weekly retail sales using historical store/department sales, calendar variables, economic variables and promotion signals.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{summary['raw_rows']:,}")
    c2.metric("Train Rows", f"{summary['train_rows']:,}")
    c3.metric("Test Rows", f"{summary['test_rows']:,}")
    c4.metric("Best Model", summary["best_model"])
    st.image(str(ROOT/"outputs/sales_trend.png"), caption="Historical weekly sales trend")
    st.info("The bundled dataset is a reproducible Walmart-schema demonstration dataset. The public Walmart forecasting dataset uses Store, Dept, Date, Weekly_Sales and IsHoliday, with additional store/features files. Replace data/walmart_sales.csv with your licensed dataset if required.")

with tabs[1]:
    st.subheader("Dataset")
    st.write("Raw data preview")
    st.dataframe(df.head(20), use_container_width=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Stores", df["Store"].nunique())
    st.write("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

with tabs[2]:
    st.subheader("Data Preprocessing")
    st.markdown("""
    **Steps applied**
    1. Convert `Date` to datetime.
    2. Sort by Store → Department → Date.
    3. Detect missing values.
    4. Fill numeric missing values with median.
    5. Create chronological train/test split.
    6. Remove rows where lag/rolling features cannot be computed.
    """)
    st.metric("Missing cells before cleaning", f"{summary['missing_cells_before']:,}")
    st.metric("Rows after preprocessing", f"{summary['clean_rows']:,}")
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing Values").to_frame(), use_container_width=True)

with tabs[3]:
    st.subheader("Feature Engineering")
    st.write("Time-series and business features used by the model:")
    st.dataframe(pd.DataFrame({"Feature":features}), use_container_width=True)
    st.markdown("""
    - **Calendar:** Year, Month, Week, Quarter
    - **Lag:** previous-week and previous-4-week sales
    - **Rolling:** 4-week mean and standard deviation
    - **Business:** Store, Department, Holiday, Temperature, Fuel Price, Markdown, CPI, Unemployment, Store Size
    """)

with tabs[4]:
    st.subheader("Machine Learning Model")
    st.write("Two regression models are compared. Random Forest is selected as the final model.")
    st.dataframe(comparison.style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","R2":"{:.4f}"}), use_container_width=True)
    st.image(str(ROOT/"outputs/model_comparison.png"), caption="RMSE comparison")
    st.image(str(ROOT/"outputs/feature_importance.png"), caption="Top feature importance")

with tabs[5]:
    st.subheader("Prediction Output")
    st.write("Actual vs predicted values on the chronological hold-out set.")
    st.image(str(ROOT/"outputs/actual_vs_predicted.png"), caption="Actual vs Predicted")
    c1,c2,c3 = st.columns(3)
    c1.metric("MAE", f"{summary['mae']:,.2f}")
    c2.metric("RMSE", f"{summary['rmse']:,.2f}")
    c3.metric("R²", f"{summary['r2']:.4f}")
    st.dataframe(pred.head(50), use_container_width=True)
    st.download_button("Download Predictions CSV", pred.to_csv(index=False), "predictions.csv", "text/csv")

with tabs[6]:
    st.subheader("🔮 Interactive Weekly Sales Prediction")
    st.write("Enter business conditions. The saved Random Forest model returns a weekly sales estimate.")
    col1,col2,col3 = st.columns(3)
    store = col1.number_input("Store", min_value=1, max_value=int(df.Store.max()), value=1)
    dept = col2.number_input("Department", min_value=1, max_value=int(df.Dept.max()), value=1)
    date = col3.date_input("Forecast Date", value=pd.Timestamp("2021-12-17").date())
    col4,col5,col6 = st.columns(3)
    holiday = col4.selectbox("Holiday Week", [0,1])
    temperature = col5.number_input("Temperature", value=float(df.Temperature.median()))
    fuel = col6.number_input("Fuel Price", value=float(df.Fuel_Price.median()))
    col7,col8,col9 = st.columns(3)
    markdown = col7.number_input("Markdown Total", value=float(df.MarkDown_Total.median()))
    cpi = col8.number_input("CPI", value=float(df.CPI.median()))
    unemployment = col9.number_input("Unemployment", value=float(df.Unemployment.median()))
    size = st.number_input("Store Size", value=float(df.Size.median()))
    lag1 = st.number_input("Previous Week Sales (Lag 1)", value=float(df.Weekly_Sales.median()))
    lag4 = st.number_input("Sales 4 Weeks Ago (Lag 4)", value=float(df.Weekly_Sales.median()))
    rollmean = st.number_input("4-Week Rolling Mean", value=float(df.Weekly_Sales.median()))
    rollstd = st.number_input("4-Week Rolling Std", value=float(df.Weekly_Sales.std()))

    if st.button("Predict Weekly Sales", type="primary"):
        d = pd.Timestamp(date)
        row = pd.DataFrame([{
            "Store":store, "Dept":dept, "IsHoliday":holiday,
            "Temperature":temperature, "Fuel_Price":fuel, "MarkDown_Total":markdown,
            "CPI":cpi, "Unemployment":unemployment, "Size":size,
            "Year":d.year, "Month":d.month, "Week":int(d.isocalendar().week),
            "Quarter":d.quarter, "Lag_1":lag1, "Lag_4":lag4,
            "Rolling_Mean_4":rollmean, "Rolling_Std_4":rollstd
        }])
        prediction = model.predict(row[features])[0]
        st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")

st.sidebar.title("Project Pipeline")
st.sidebar.write("Dataset → Preprocessing → Feature Engineering → Model → Evaluation → Prediction")
st.sidebar.caption("Portfolio project • 2026")
