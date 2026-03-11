# crude-oil-price-forecaster
A Streamlit-based web application using Random Forest and Time-Series models to forecast global crude oil spot prices based on fundamental market indicators.
## 📌 Project Overview
Predicting crude oil prices is a complex challenge influenced by supply, demand, and geopolitical factors. This application provides an interactive platform to visualize historical trends and predict future prices using a machine learning pipeline.

Fundamental Regression: Predicts country-specific spot prices using market benchmarks (Brent, WTI, OPEC).

Time-Series Forecasting: Implements historical lag analysis to project prices up to 36 months into the future.

Interactive Controls: Sidebar selection for countries, model types, and forecast horizons.

Automated Evaluation: Generates real-time Mean Squared Error (MSE) and Squared Error metrics.

Data Export: Automatically saves predictions and error logs to a structured Excel file.

## 🛠️ Tech Stack
Frontend: Streamlit

Machine Learning: Scikit-Learn (Random Forest Regressor, Pipeline, OneHotEncoder)

Data Processing: Pandas, NumPy

Visualization: Matplotlib

Storage: Openpyxl (Excel Integration)

## 🚀 How to run this project ?
Follow these steps to run the project on your local machine:

1. Clone the Repository:


git clone https://github.com/YOUR_USERNAME/crude-oil-price-forecaster.git
cd crude-oil-price-forecaster

2. Install Dependencies:


pip install -r requirements.txt

3. Run the Application:

streamlit run app.py


## 📁 File Structure

crude-oil-price-forecaster/

├── app.py

├── requirements.txt

├── crude_oil_with_predictions.xlsx

├── crude_oil_with_predictions_with_mse.xlsx

├── forecast.xlsx

├── README.md

└── .gitignore
## 📊 Model Methodology
The application uses a Random Forest Regressor wrapped in a Pipeline. This ensures that categorical data like Country and Region are properly encoded without data leakage.

Indicators Used: Brent Price, WTI Price, OPEC Basket, Production (kbd), Exports, Consumption, Rig Count, and Geo-Political Risk Index.

Target Variable: Country_Spot_Price_USD.

Validation: The model uses backtesting on historical data to calculate the MSE before generating future forecasts.

## ✍️ Author & Contact

Name: Sai Shyam Achari

GitHub: https://github.com/sai-shyamachari

LinkedIn: www.linkedin.com/in/sai-shyam-achari-5870a0373

Email: saishyamachari@gmail.com
