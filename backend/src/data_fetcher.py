import requests
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# NewsAPI configuration
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = "891f527ee0624fb48d71a312220cfea0"  # Your API key

def fetch_stock_data(ticker, start_date, end_date, api_key=None):
    """
    Fetch historical stock data using Yahoo Finance as primary source
    """
    try:
        # Clean the ticker symbol (remove .US if present)
        ticker = ticker.replace('.US', '')
        
        logger.info(f"Fetching Yahoo Finance data for {ticker} from {start_date} to {end_date}")
        
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            logger.error(f"No data returned from Yahoo Finance for {ticker}")
            return None, None
            
        # Convert to required format
        df = df[['Close']].rename(columns={'Close': 'close'})
        df.index = pd.to_datetime(df.index)
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        return df['close'].values.reshape(-1, 1), df.index.tolist()
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

def fetch_news_articles(ticker, days_back=30, api_key=NEWSAPI_KEY):
    """
    Fetch financial news articles for a specific stock ticker using NewsAPI
    """
    try:
        if not api_key:
            logger.warning("NewsAPI key not provided - news sentiment disabled")
            return []
            
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for NewsAPI
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Prepare query - search for ticker in business/finance news
        query = f"{ticker} AND (stock OR stocks OR market OR finance OR financial)"
        
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'pageSize': 100,  # Max allowed per request
            'domains': 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,wsj.com'
        }
        
        logger.info(f"Fetching news articles for {ticker} from {from_date} to {to_date}")
        response = requests.get(NEWSAPI_URL, params=params)
        response.raise_for_status()
        
        articles = response.json().get('articles', [])
        logger.info(f"Found {len(articles)} news articles for {ticker}")
        
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {ticker}: {str(e)}")
        return []

def analyze_sentiment(articles):
    """
    Perform sentiment analysis on news articles using TextBlob
    """
    if not articles:
        # Return neutral sentiment if no articles
        return np.zeros(1).reshape(-1, 1)
    
    sentiment_scores = []
    for article in articles:
        try:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if not text:
                sentiment_scores.append(0.0)
                continue
                
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity)
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for article: {str(e)}")
            sentiment_scores.append(0.0)
    
    return np.array(sentiment_scores).reshape(-1, 1)