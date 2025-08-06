import pandas as pd
import yfinance as yf
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logger
import os
import tempfile
import prophet
import numpy as np
import logging
# Verify Prophet version
try:
    logger.info(f"Prophet version: {prophet.__version__}")
except AttributeError:
    logger.error("Prophet module does not have __version__ attribute. Reinstall prophet.")
    raise ImportError("Invalid prophet installation.")

# Set custom temporary directory
tempfile.tempdir = r"C:\Investro\tmp"
os.makedirs(tempfile.tempdir, exist_ok=True)
os.environ['TMPDIR'] = r"C:\Investro\tmp"  # Ensure Stan uses custom temp directory

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Fetch historical stock price data with retry
ticker = 'AAPL'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_stock_data(ticker, start, end):
    logger.info(f"Fetching stock data for {ticker}")
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    if hist.empty:
        raise ValueError(f"No stock price data retrieved for {ticker}.")
    logger.info(f"Retrieved {len(hist)} rows for {ticker}")
    return hist

try:
    hist = fetch_stock_data(ticker, start='2020-01-01', end='2025-04-30')
except Exception as e:
    logger.error(f"Error fetching stock data: {e}")
    raise

# Create stock price DataFrame with timezone-naive ds
df = pd.DataFrame({
    'ds': hist.index.tz_localize(None),  # Remove timezone
    'y': hist['Close']
}).reset_index(drop=True)
df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime64[ns]

# Step 2: Fetch news sentiment data via Flask proxy or NewsAPI.in
use_proxy = True  # Set to False to use NewsAPI.in
if use_proxy:
    url = 'http://localhost:5000/news/Apple'
else:
    newsapi_in_key = '4f25485a971541e9af6be414766541a3'  # Replace with NewsAPI.in key
    url = f'https://newsapi.in/v2/everything?q=Apple&from=2020-01-01&to=2025-04-30&apiKey={newsapi_in_key}'

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if use_proxy and data.get('status') == 'error':
        logger.warning(f"Flask proxy returned error: {data.get('message')}")
        articles = data.get('articles', [])
    else:
        articles = data.get('articles', [])
    if not articles:
        logger.warning("No news articles retrieved.")
        articles = [
            {'publishedAt': '2025-04-01T00:00:00Z', 'title': 'Apple reports record-breaking earnings.', 'description': ''},
            {'publishedAt': '2025-04-03T00:00:00Z', 'title': 'Apple faces supply chain issues.', 'description': ''},
            {'publishedAt': '2025-04-05T00:00:00Z', 'title': 'Apple launches new product.', 'description': ''},
            {'publishedAt': '2025-04-07T00:00:00Z', 'title': 'Apple stock surges.', 'description': ''}
        ]
except Exception as e:
    logger.error(f"Error fetching news data: {e}")
    articles = [
        {'publishedAt': '2025-04-01T00:00:00Z', 'title': 'Apple reports record-breaking earnings.', 'description': ''},
        {'publishedAt': '2025-04-03T00:00:00Z', 'title': 'Apple faces supply chain issues.', 'description': ''},
        {'publishedAt': '2025-04-05T00:00:00Z', 'title': 'Apple launches new product.', 'description': ''},
        {'publishedAt': '2025-04-07T00:00:00Z', 'title': 'Apple stock surges.', 'description': ''}
    ]

analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []
for article in articles:
    date = article['publishedAt'][:10]
    text = article.get('title', '') + ' ' + article.get('description', '')
    score = analyzer.polarity_scores(text)['compound']
    sentiment_scores.append({'ds': date, 'sentiment': score})

sentiment_df = pd.DataFrame(sentiment_scores)
sentiment_df['ds'] = pd.to_datetime(sentiment_df['ds'])  # Ensure datetime64[ns]
sentiment_df = sentiment_df.groupby('ds').mean().reset_index()  # Average sentiment per day

# Step 3: Merge sentiment with stock data
try:
    df = df.merge(sentiment_df, on='ds', how='left')
except ValueError as e:
    logger.error(f"Merge error: {e}")
    raise
df['sentiment'] = df['sentiment'].fillna(0)  # Neutral for days without news

# Step 4: Normalize sentiment regressor
scaler = StandardScaler()
df['sentiment'] = scaler.fit_transform(df[['sentiment']])
df['sentiment'] = df['sentiment'].clip(-3, 3)  # Clip extreme values

# Step 5: Validate DataFrame
logger.info(f"DataFrame shape: {df.shape}")
logger.info(f"DataFrame head:\n{df.head(10)}")
logger.info(f"NaN counts:\n{df.isna().sum()}")
logger.info(f"Non-NaN rows: {df.dropna().shape[0]}")
logger.info(f"ds dtype: {df['ds'].dtype}")
logger.info(f"y stats: min={df['y'].min()}, max={df['y'].max()}, mean={df['y'].mean()}, std={df['y'].std()}")
logger.info(f"sentiment stats: min={df['sentiment'].min()}, max={df['sentiment'].max()}, mean={df['sentiment'].mean()}, std={df['sentiment'].std()}")
logger.info(f"ds gaps:\n{df['ds'].diff().dropna().value_counts()}")
if df.dropna().shape[0] < 100:
    logger.error("DataFrame has fewer than 100 non-NaN rows, which may cause optimization failure.")
    raise ValueError("DataFrame has fewer than 100 non-NaN rows.")
if df['y'].isna().any():
    df['y'] = df['y'].fillna(method='ffill')  # Fill missing prices
if df['sentiment'].isna().any():
    df['sentiment'] = df['sentiment'].fillna(0)  # Ensure no NaN in sentiment
df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()  # Remove infinities
if df['y'].std() < 0.01:
    logger.error("y values have low variance (std < 0.01), which may cause optimization failure.")
    raise ValueError("y values have insufficient variation.")
if df['sentiment'].std() < 0.01:
    logger.warning("sentiment values have low variance (std < 0.01), using synthetic sentiment.")
    df['sentiment'] = np.random.normal(0, 1, len(df))  # Fallback to synthetic sentiment
# Ensure daily frequency
df = df.sort_values('ds').set_index('ds').asfreq('D', method='ffill').reset_index()

# Step 6: Train Prophet model
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=False,
    weekly_seasonality=False,
    growth='linear',
    changepoint_prior_scale=0.05,
    stan_backend='CMDSTANPY'
)
model.add_regressor('sentiment', standardize=False)
try:
    model.fit(df)
except Exception as e:
    logger.error(f"Error training Prophet model: {e}")
    raise

# Step 7: Create future dataframe
future = model.make_future_dataframe(periods=30)
future['sentiment'] = scaler.transform([[df['sentiment'].iloc[-1]]])[0]

# Step 8: Make predictions
forecast = model.predict(future)

# Step 9: Evaluate model
mae = mean_absolute_error(df['y'], forecast['yhat'][:len(df)])
logger.info(f"MAE: {mae}")

# Step 10: Prepare output for Streamlit
forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
forecast_output.to_csv('forecast_with_sentiment.csv', index=False)