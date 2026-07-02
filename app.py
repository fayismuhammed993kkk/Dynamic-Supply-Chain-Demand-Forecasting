import streamlit as st
import joblib
import pandas as pd
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rossmann Sales Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Rossmann Store Sales Forecasting")
st.write(
    "Enter the store details below to predict the daily sales using our optimized XGBoost engine."
)

# -----------------------------
# Sidebar Model Metadata
# -----------------------------
with st.sidebar:
    st.header("🏆 Model Performance")
    st.markdown("**Engine:** XGBoost Regressor")
    st.markdown("**Accuracy ($R^2$):** 88.21%")
    st.markdown("**Avg. Error (MAE):** $757.67")
    st.write("---")
    st.caption("This tool forecasts daily retail revenue based on historical store structures and promotional calendars.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_sales_model.pkl")

model = load_model()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🏬 Store Information")

col1, col2 = st.columns(2)

with col1:
    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
    
    # Human-readable mapping for Store Type
    type_labels = {'a': 'Type A (Standard)', 'b': 'Type B (Flagship)', 'c': 'Type C (Medium)', 'd': 'Type D (Extended)'}
    store_type = st.selectbox("Store Type", options=list(type_labels.keys()), format_func=lambda x: type_labels[x])
    
    # Human-readable mapping for Assortment
    assort_labels = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}
    assortment = st.selectbox("Assortment Level", options=list(assort_labels.keys()), format_func=lambda x: assort_labels[x])
    
    # Human-readable day names mapped to numeric IDs
    day_labels = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    day_of_week = st.selectbox("Day of Week", options=list(day_labels.keys()), format_func=lambda x: day_labels[x], index=0)

    # Open/Closed toggle to handle zero-sales scenarios gracefully
    is_open = st.selectbox("Is the Store Open Today?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No (Force $0 Sales)")

with col2:
    promo = st.selectbox("Running Active Promo?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    promo2 = st.selectbox("Continuous Promo2 Active?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    competition_distance = st.number_input("Competition Distance (Meters)", min_value=0, value=1000)
    competition_open_month = st.number_input(
    "Competition Open Since Month",
    min_value=1,
    max_value=12,
    value=1
)
    competition_open_year = st.number_input(
    "Competition Open Since Year",
    min_value=1900,
    max_value=2025,
    value=2010
)

st.subheader("📅 Promotion & Calendar Context")

col3, col4 = st.columns(2)

with col3:
    if promo2 == 1:
    promo2_since_week = st.number_input(
        "Promo2 Active Since Week",
        min_value=1,
        max_value=52,
        value=1
    )
else:
    promo2_since_week = 0
    if promo2 == 1:
    promo2_since_year = st.number_input(
        "Promo2 Active Since Year",
        min_value=2008,
        max_value=2025,
        value=2013
    )
else:
    promo2_since_year = 0
    
    
    # Clean, human-readable holiday mapping to replace abstract letters
    holiday_mapping = {
        '0': 'Regular Day (No Holiday)',
        'a': 'Public Holiday',
        'b': 'Easter Holiday',
        'c': 'Christmas Season'
    }
    state_holiday = st.selectbox("State Holiday Context", options=list(holiday_mapping.keys()), format_func=lambda x: holiday_mapping[x])

with col4:
    promo_interval = st.selectbox("Promo Interval Cycle", ["None", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"])
    school_holiday = st.selectbox("School Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    selected_date = st.date_input("Select Target Date")

# -----------------------------
# Extract Date Features
# -----------------------------
date = pd.to_datetime(selected_date)
year = date.year
month = date.month
day = date.day
weekofyear = int(date.isocalendar().week)
is_weekend = 1 if date.weekday() >= 5 else 0

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🚀 Forecast Daily Sales", type="primary"):
    
    # If store is marked closed, bypass the pipeline to guarantee $0 output safely
    if is_open == 0:
        st.warning("Store is marked as Closed.")
        st.metric(label="📈 Predicted Daily Sales", value="$0.00")
    else:
        input_data = pd.DataFrame({
            "Store": [store_id],
            "DayOfWeek": [day_of_week],
            "Promo": [promo],
            "Open": [is_open],
            "StateHoliday": [state_holiday],
            "SchoolHoliday": [school_holiday],
            "StoreType": [store_type],
            "Assortment": [assortment],
            "CompetitionDistance": [competition_distance],
            "CompetitionOpenSinceMonth": [competition_open_month],
            "CompetitionOpenSinceYear": [competition_open_year],
            "Promo2": [promo2],
            "Promo2SinceWeek": [promo2_since_week],
            "Promo2SinceYear": [promo2_since_year],
            "PromoInterval": [promo_interval],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "WeekOfYear": [weekofyear],
            "IsWeekend": [is_weekend]
        })

        # Run inference through pipeline
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        st.success("Prediction generated successfully!")
        st.metric(
            label="📈 Predicted Daily Sales",
            value=f"${prediction:,.2f}"
        )