# 📈 Dynamic Supply Chain Demand Forecasting using Machine Learning

`Python` `Xgboost` `scikit-learn` `Streamlit` `demand-forecasting` `supply-chain` `Machine Learning`

A machine learning-based retail sales forecasting system that predicts daily store sales using historical sales data, store characteristics, promotions, holidays, competition information, and time-based features.

## 🚀 Live Demo
https://github.com/user-attachments/assets/3cc0dbac-864d-45ad-9272-6640f1694dec

🌐 **Streamlit App:**
https://dynamic-demand-forecasting.streamlit.app/

---

## 📌 Project Overview

Accurate demand forecasting helps retailers make better decisions about inventory, staffing, and promotional planning.

This project develops a **Retail Sales Forecasting system** using Machine Learning to predict daily sales for retail stores.

The project covers the complete Machine Learning workflow:

**Data Collection → Data Preprocessing → EDA → Feature Engineering → Model Training → Model Evaluation → Deployment**

---

## 🎯 Objective

The main objectives of this project are to:

* Predict daily retail store sales.
* Identify important factors affecting sales.
* Compare different regression algorithms.
* Select the best-performing model based on evaluation metrics.
* Deploy the final model as an interactive Streamlit application.
* Support better demand and inventory planning.

---

## 📊 Dataset

* **Dataset:** Rossmann Store Sales
* **Source:** Kaggle
* **Final Dataset Size:** 844,338 rows
* **Features:** 18
* **Target Variable:** `Sales`

The dataset contains information related to:

* Store characteristics
* Promotions
* Holidays
* Competition
* Product assortment
* Historical sales
* Date and seasonal patterns

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**

---

## 🔄 Machine Learning Workflow

### 1. Data Preprocessing

* Merged `sales.csv` and `store.csv` using the `Store` column.
* Handled missing values.
* Removed the `Customers` feature to avoid data leakage during forecasting.
* Filtered records where stores were open and sales were greater than zero.
* Converted the `Date` column into a datetime format.

### 2. Feature Engineering

Created additional time-based features:

* `Year`
* `Month`
* `Day`
* `WeekOfYear`
* `IsWeekend`

### 3. Categorical Encoding

Categorical variables were processed using:

`ColumnTransformer` + `OneHotEncoder`

This converts categorical features such as `StoreType`, `Assortment`, and `PromoInterval` into numerical features suitable for Machine Learning models.

---

## 📊 Exploratory Data Analysis

Several analyses were performed to understand sales patterns:

* Sales distribution
* Average monthly sales
* Average sales by day of week
* Sales during promotional and non-promotional periods
* Sales distribution across store types
* Numerical feature correlation analysis
* XGBoost feature importance

### Key Observations

* Sales show noticeable variation across different months.
* Promotional periods are associated with higher average sales.
* Sales patterns vary across different store types.
* Store type and promotion-related features were among the important predictors in the final XGBoost model.

---

## 🤖 Models Compared

Four regression algorithms were trained and evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **XGBoost Regressor**

The models were evaluated using:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R² Score**

---

## 🏆 Final Model Performance

| Metric       | XGBoost |
| ------------ | ------: |
| **MAE**      |  757.67 |
| **RMSE**     | 1066.65 |
| **R² Score** |  0.8821 |

The **XGBoost Regressor** achieved the best overall performance among the models evaluated and was selected as the final model.

### R² Score: 88.21%

The model explains approximately **88.21% of the variation in sales** on the test data.

---

## 🔍 Important Features

Feature importance analysis was performed using the trained XGBoost model.

Some of the important features included:

* `StoreType_b`
* `Promo`
* `Promo2SinceYear`
* `CompetitionDistance`
* `PromoInterval`
* `Promo2`
* `CompetitionOpenSinceYear`
* `CompetitionOpenSinceMonth`
* `Assortment`
* `Store`

This helps provide insight into the factors influencing the model's sales predictions.

---

## 🌐 Streamlit Application

The trained XGBoost model was deployed using **Streamlit**.

The application allows users to provide store-related inputs and receive an estimated daily sales prediction.

### Application Workflow

**User Input → Preprocessing → Trained XGBoost Model → Sales Prediction**

The trained model is saved using **Joblib** and loaded by the Streamlit application during prediction.

---

## 📁 Project Structure

```text
Dynamic-Supply-Chain-Demand-Forecasting/
│
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── sales.csv
│   └── store.csv
│
├── models/
│   └── xgboost_sales_model.pkl
│
├── notebooks/
    └── Retail_Sales_Forecasting.ipynb
```

---

## 💡 Business Applications

The forecasting system can support:

* 📦 Inventory planning
* 🏪 Store-level demand planning
* 👥 Staff allocation
* 📢 Promotional planning
* 📊 Sales analysis
* 🚚 Supply chain decision-making

Accurate demand predictions can help businesses reduce the risk of **overstocking and understocking** and make more informed operational decisions.

---

## 👨‍💻 Author

**Muhammed Fayis**

*Data Science Machine Learning* 
