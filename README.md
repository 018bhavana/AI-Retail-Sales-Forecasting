# AI-Driven Retail Sales Forecasting — Web App

## What this project contains
A complete Streamlit webpage showing:
- Dataset preview and statistics
- Data preprocessing
- Missing-value handling
- Feature engineering
- Linear Regression vs Random Forest comparison
- Feature importance
- Actual vs predicted output
- MAE, RMSE and R²
- Interactive weekly-sales prediction form
- CSV download of predictions

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset note
The bundled CSV is a reproducible demonstration dataset using the same core Walmart forecasting schema. Public Walmart forecasting datasets commonly contain `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`, plus store and regional/features tables. The project can be switched to the original dataset by replacing `data/walmart_sales.csv` and adapting columns if needed.

## Suggested resume title
AI-Driven Retail Sales Forecasting | Python, Pandas, NumPy, Scikit-learn, Streamlit

## Resume bullets
- Built an end-to-end retail sales forecasting application with data preprocessing, time-series feature engineering and machine learning.
- Compared Linear Regression and Random Forest models using MAE, RMSE and R², selecting Random Forest as the final model.
- Developed an interactive Streamlit dashboard for dataset exploration, model evaluation, visual analytics and weekly sales prediction.
