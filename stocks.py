import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import warnings
from datetime import datetime, timedelta


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch stock data with error handling
def fetch_stock_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        if stock_data.empty:
            st.error(f"No data found for ticker {ticker}. Please check if the ticker symbol is correct.")
            return None
            
        stock_data = stock_data[['Close']].reset_index()
        stock_data.columns = ['ds', 'y']  # Rename directly for Prophet
        stock_data['ds'] = pd.to_datetime(stock_data['ds']).dt.tz_localize(None)
        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Make predictions with improved error handling
def predict_stock_trends(data, periods=30):
    try:
        if data is None or len(data) < 30:  # Minimum data requirement
            st.error("Insufficient data for prediction. Please ensure at least 30 data points.")
            return None, None, None, None, None, None

        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Check for and handle outliers
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['y'] < (Q1 - 1.5 * IQR)) | (df['y'] > (Q3 + 1.5 * IQR)))]
        
        # Log transform the data
        df['y'] = np.log(df['y'].clip(lower=0.01))
        
        # Configure Prophet with more stable parameters
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            n_changepoints=25
        )
        
        
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') 
            model.fit(df)
        
       
        future_dates = pd.date_range(
            start=df['ds'].max() + timedelta(days=1),
            periods=periods,
            freq='B'
        )
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = model.predict(future)
        
        # Inverse transform predictions
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = np.exp(forecast[col])
        
        # Calculate error metrics
        historical_forecast = model.predict(df[['ds']])
        historical_forecast['yhat'] = np.exp(historical_forecast['yhat'])
        actual = np.exp(df['y'])
        predicted = historical_forecast['yhat']
        
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return forecast, model, mae, mse, rmse, mape
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None, None, None

def main():
    st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
    
    st.title("📈 INvestro - Stock Prediction App")
    
    # Add description
    st.markdown("""
    This app predicts stock price trends using Facebook's Prophet model. 
    Please note that predictions are for educational purposes only.
    """)
    
    # Input parameters
    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Stock Ticker", "RELIANCE.NS")
        
        # Limit date range to reduce potential issues
        max_date = datetime.now()
        min_date = max_date - timedelta(days=365*2)  # 2 years max
        
        start_date = st.date_input(
            "Start Date", 
            min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date", 
            max_date,
            min_value=start_date,
            max_value=max_date
        )
        
        periods = st.slider("Prediction Days", 5, 60, 30)  # Reduced max prediction period
        
    if st.sidebar.button("Predict"):
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(ticker, start_date, end_date)
            
        if data is not None:
            st.subheader(f"Historical Data for {ticker}")
            st.dataframe(data.tail(), height=200)
            
            # Historical price plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Historical Price'))
            fig.update_layout(title="Historical Stock Price", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                result = predict_stock_trends(data, periods)
                
            if all(x is not None for x in result):
                forecast, model, mae, mse, rmse, mape = result
                
                # Display predictions
                st.subheader("Predictions")
                prediction_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                prediction_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                st.dataframe(prediction_df.tail(), height=200)
                
                # Prediction plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                        mode='lines', name='Predicted Price'))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                        fill=None, mode='lines', name='Lower Bound'))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                        fill='tonexty', mode='lines', name='Upper Bound'))
                fig2.update_layout(title="Price Prediction", xaxis_title="Date", 
                                 yaxis_title="Price", hovermode='x unified')
                st.plotly_chart(fig2)
                
                # Error metrics
                st.subheader("Model Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAPE", f"{mape:.2f}%")
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("MSE", f"{mse:.2f}")

if __name__ == "__main__":
    main()