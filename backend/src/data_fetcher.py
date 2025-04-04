import requests
import pandas as pd
import numpy as np
from textblob import TextBlob

def fetch_stock_data(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data using the EODHD API.
    """
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}"
    params = {
        'api_token': api_key,
        'from': start_date,
        'to': end_date,
        'fmt': 'json'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df['close'].values.reshape(-1, 1), df.index
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock data: {e}")
        return None, None

def fetch_news_articles(api_key, ticker):
    """
    Fetch financial news articles using the EODHD API.
    """
    url = f"https://eodhistoricaldata.com/api/news"
    params = {
        'api_token': api_key,
        's': ticker,
        'limit': 100
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment(articles):
    """
    Perform sentiment analysis on news articles.
    """
    sentiment_scores = []
    for article in articles:
        text = article['title'] + " " + article['content']
        analysis = TextBlob(text)
        sentiment_scores.append(analysis.sentiment.polarity)
    return np.array(sentiment_scores).reshape(-1, 1)