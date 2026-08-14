
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AI Retail Sales Forecasting", page_icon="📈", layout="wide")

# ============================================================
# DATASET IS CREATED INSIDE THE WEB APP — NO CSV UPLOAD NEEDED
# ============================================================
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=104, freq="W-FRI")
    rows = []

    for store in range(1, 21):
        store_factor = 0.72 + rng.integers(50000, 220000) / 400000
        for dept in range(1, 11):
            dept_factor = 0.65 + dept / 18
            base = 9000 * store_factor * dept_factor

            for date in dates:
                week = int(date.isocalendar().week)
                month = date.month
                holiday = int(
                    (month == 11 and week >= 47) or
                    (month == 12 and week >= 50) or
                    (month == 7 and week == 27)
                )

                seasonal = (
                    1
                    + 0.18 * np.sin(2*np.pi*week/52)
                    + 0.08 * np.cos(2*np.pi*week/26)
                )
                holiday_boost = 1.28 if holiday else 1.0

                temperature = 55 + 25*np.sin(2*np.pi*(month-1)/12) + rng.normal(0, 4)
                fuel = 2.4 + 0.012*(date.year-2021) + rng.normal(0, .06)
                cpi = 210 + 0.35*((date-dates[0]).days/30) + rng.normal(0, .4)
                unemployment = 6 + rng.normal(0, .25)
                markdown = max(0, rng.normal(1200 if holiday else 300, 500))

                sales = base * seasonal * holiday_boost
                sales *= (1 + markdown/18000)
                sales *= (1 + 0.003*(temperature-55))
                sales += rng.normal(0, base*0.10)

                rows.append([
                    store, dept, date, max(0, sales), holiday,
                    temperature, fuel, markdown, cpi, unemployment,
                    int(store_factor * 400000)
                ])

    return pd.DataFrame(rows, columns=[
        "Store","Dept","Date","Weekly_Sales","IsHoliday",
        "Temperature","Fuel_Price","MarkDown_Total",
        "CPI","Unemployment","Size"
    ])

@st.cache_data
def prepare_data(df):
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    for col in ["Temperature","Fuel_Price","MarkDown_Total","CPI","Unemployment"]:
        data[col] = data[col].fillna(data[col].median())

    data = data.sort_values(["Store","Dept","Date"]).reset_index(drop=True)

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter

    group = data.groupby(["Store","Dept"])["Weekly_Sales"]
    data["Lag_1"] = group.shift(1)
    data["Lag_4"] = group.shift(4)
    data["Rolling_Mean_4"] = data.groupby(
        ["Store","Dept"]
    )["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )
    data["Rolling_Std_4"] = data.groupby(
        ["Store","Dept"]
    )["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(4).std()
    )

    return data.dropna().reset_index(drop=True)

@st.cache_resource
def train_models(data):
    features = [
        "Store","Dept","IsHoliday","Temperature","Fuel_Price",
        "MarkDown_Total","CPI","Unemployment","Size","Year",
        "Month","Week","Quarter","Lag_1","Lag_4",
        "Rolling_Mean_4","Rolling_Std_4"
    ]

    cut = int(len(data) * .80)
    train = data.iloc[:cut]
    test = data.iloc[cut:]

    lr = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    lr.fit(train[features], train["Weekly_Sales"])
    rf.fit(train[features], train["Weekly_Sales"])

    lr_pred = lr.predict(test[features])
    rf_pred = rf.predict(test[features])

    return features, train, test, lr, rf, lr_pred, rf_pred

# ------------------------------------------------------------
# RUN DATASET + PREPROCESSING + MODEL AUTOMATICALLY
# ------------------------------------------------------------
raw = create_dataset()
data = prepare_data(raw)
features, train, test, lr, rf, lr_pred, rf_pred = train_models(data)

def get_metrics(y, p):
    return (
        mean_absolute_error(y,p),
        np.sqrt(mean_squared_error(y,p)),
        r2_score(y,p)
    )

lr_mae, lr_rmse, lr_r2 = get_metrics(test["Weekly_Sales"], lr_pred)
rf_mae, rf_rmse, rf_r2 = get_metrics(test["Weekly_Sales"], rf_pred)

# ============================================================
# WEBPAGE
# ============================================================
st.title("📈 AI-Driven Retail Sales Forecasting")
st.caption("Complete Machine Learning Project — Dataset, Preprocessing, Features, Model & Prediction in ONE Webpage")

st.success(
    "✅ No CSV upload required. The dataset is generated inside this webpage, "
    "preprocessed automatically, the model is trained automatically, and predictions "
    "can be made below."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "📊 Dataset",
    "🧹 Preprocessing",
    "🧩 Features",
    "🤖 Model & Output",
    "🔮 Predict"
])

with tab1:
    st.header("Project Pipeline")
    st.markdown("""
    **Dataset → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction**

    ### Objective
    Predict weekly retail sales based on historical sales, store/department information,
    holidays, economic variables and time-series patterns.
    """)

    a,b,c,d = st.columns(4)
    a.metric("Dataset Rows", f"{len(raw):,}")
    b.metric("Training Rows", f"{len(train):,}")
    c.metric("Testing Rows", f"{len(test):,}")
    d.metric("Best Model", "Random Forest")

    st.subheader("Sales Trend")
    trend = raw.groupby("Date")["Weekly_Sales"].sum()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(trend.index, trend.values)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Historical Weekly Sales")
    st.pyplot(fig)

with tab2:
    st.header("📊 Dataset")
    st.write("The dataset is available directly inside the application — no file selection is needed.")
    st.dataframe(raw.head(50), use_container_width=True)

    a,b,c = st.columns(3)
    a.metric("Rows", f"{len(raw):,}")
    b.metric("Columns", len(raw.columns))
    c.metric("Stores", raw["Store"].nunique())

    st.subheader("Dataset Statistics")
    st.dataframe(raw.describe().T, use_container_width=True)

with tab3:
    st.header("🧹 Data Preprocessing")

    st.markdown("""
    ### Steps
    1. Date conversion
    2. Sorting by Store, Department and Date
    3. Missing-value handling using median
    4. Time-based feature extraction
    5. Lag feature creation
    6. Rolling-window feature creation
    7. Chronological 80/20 train-test split
    """)

    st.write("Missing values before preprocessing:")
    st.dataframe(raw.isna().sum().rename("Missing Values").to_frame())

    st.metric("Rows after preprocessing", f"{len(data):,}")

with tab4:
    st.header("🧩 Feature Engineering")

    feature_info = pd.DataFrame({
        "Feature": features,
        "Purpose": [
            "Store identifier",
            "Department identifier",
            "Holiday indicator",
            "Temperature",
            "Fuel price",
            "Markdown / promotion amount",
            "Consumer price index",
            "Unemployment",
            "Store size",
            "Year",
            "Month",
            "Week number",
            "Quarter",
            "Previous week sales",
            "Sales four weeks ago",
            "Previous 4-week average sales",
            "Previous 4-week sales volatility"
        ]
    })
    st.dataframe(feature_info, use_container_width=True)

with tab5:
    st.header("🤖 Model Training & Evaluation")

    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [lr_mae, rf_mae],
        "RMSE": [lr_rmse, rf_rmse],
        "R²": [lr_r2, rf_r2]
    })

    st.dataframe(
        comparison.style.format({
            "MAE":"{:,.2f}",
            "RMSE":"{:,.2f}",
            "R²":"{:.4f}"
        }),
        use_container_width=True
    )

    a,b,c = st.columns(3)
    a.metric("Random Forest MAE", f"{rf_mae:,.2f}")
    b.metric("Random Forest RMSE", f"{rf_rmse:,.2f}")
    c.metric("Random Forest R²", f"{rf_r2:.4f}")

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")
    st.pyplot(fig)

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(test["Weekly_Sales"].head(250).to_numpy(), label="Actual")
    ax.plot(rf_pred[:250], label="Predicted")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    st.pyplot(fig)

with tab6:
    st.header("🔮 Predict Weekly Retail Sales")
    st.write("Enter the values below. The trained Random Forest model will predict weekly sales.")

    col1,col2,col3 = st.columns(3)
    store = col1.number_input("Store", 1, int(raw.Store.max()), 1)
    dept = col2.number_input("Department", 1, int(raw.Dept.max()), 1)
    date = col3.date_input("Forecast Date")

    col1,col2,col3 = st.columns(3)
    holiday = col1.selectbox("Holiday Week", [0,1])
    temperature = col2.number_input("Temperature", value=float(raw.Temperature.median()))
    fuel = col3.number_input("Fuel Price", value=float(raw.Fuel_Price.median()))

    col1,col2,col3 = st.columns(3)
    markdown = col1.number_input("Markdown Total", value=float(raw.MarkDown_Total.median()))
    cpi = col2.number_input("CPI", value=float(raw.CPI.median()))
    unemployment = col3.number_input("Unemployment", value=float(raw.Unemployment.median()))

    col1,col2,col3 = st.columns(3)
    size = col1.number_input("Store Size", value=float(raw.Size.median()))
    lag1 = col2.number_input("Previous Week Sales", value=float(raw.Weekly_Sales.median()))
    lag4 = col3.number_input("Sales 4 Weeks Ago", value=float(raw.Weekly_Sales.median()))

    col1,col2 = st.columns(2)
    rollmean = col1.number_input("Previous 4-Week Average Sales", value=float(raw.Weekly_Sales.median()))
    rollstd = col2.number_input("Previous 4-Week Sales Std", value=float(raw.Weekly_Sales.std()))

    if st.button("🚀 Predict Sales", type="primary"):
        d = pd.Timestamp(date)

        row = pd.DataFrame([{
            "Store": store,
            "Dept": dept,
            "IsHoliday": holiday,
            "Temperature": temperature,
            "Fuel_Price": fuel,
            "MarkDown_Total": markdown,
            "CPI": cpi,
            "Unemployment": unemployment,
            "Size": size,
            "Year": d.year,
            "Month": d.month,
            "Week": int(d.isocalendar().week),
            "Quarter": d.quarter,
            "Lag_1": lag1,
            "Lag_4": lag4,
            "Rolling_Mean_4": rollmean,
            "Rolling_Std_4": rollstd
        }])

        prediction = rf.predict(row[features])[0]

        st.success(f"### Predicted Weekly Sales: ${prediction:,.2f}")

        st.info(
            "This prediction is generated by the Random Forest model trained "
            "inside this webpage using the built-in retail sales dataset."
        )

st.sidebar.header("ML Pipeline")
st.sidebar.write("1️⃣ Dataset")
st.sidebar.write("2️⃣ Preprocessing")
st.sidebar.write("3️⃣ Feature Engineering")
st.sidebar.write("4️⃣ Model Training")
st.sidebar.write("5️⃣ Evaluation")
st.sidebar.write("6️⃣ Prediction")
st.sidebar.success("Everything runs inside the app.")
