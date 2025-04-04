import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(stock_prices, sentiment_scores, lookback=60):
    """
    Preprocess stock prices and sentiment scores for the LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(stock_prices)
    
    X, y = [], []
    for i in range(lookback, len(scaled_prices)):
        X.append(np.hstack((scaled_prices[i-lookback:i], sentiment_scores[i-lookback:i])))
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler