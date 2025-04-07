import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def preprocess_data(stock_prices, sentiment_scores, lookback=60):
    """
    Preprocess stock prices and sentiment scores for the LSTM model.
    Returns: (X, y, scaler)
    """
    try:
        logger.info("Preprocessing data...")
        
        # Ensure inputs are numpy arrays with correct dimensions
        stock_prices = np.array(stock_prices).reshape(-1, 1)  # Ensure 2D (n_samples, 1)
        sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)  # Ensure 2D (n_samples, 1)
        
        # Validate inputs
        if len(stock_prices) != len(sentiment_scores):
            raise ValueError(f"Stock prices (len={len(stock_prices)}) and sentiment scores (len={len(sentiment_scores)}) must have same length")
            
        if len(stock_prices) < lookback:
            raise ValueError(f"Need at least {lookback} data points, got {len(stock_prices)}")
        
        # Scale stock prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(stock_prices)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            # Stack prices and sentiment scores for the lookback window
            price_window = scaled_prices[i-lookback:i]
            sentiment_window = sentiment_scores[i-lookback:i]
            
            # Combine features (price and sentiment)
            combined_features = np.hstack((price_window, sentiment_window))
            X.append(combined_features)
            
            # Target is next day's price
            y.append(scaled_prices[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Validate shapes
        if X.ndim != 3:
            raise ValueError(f"X should be 3D array (samples, timesteps, features), got {X.ndim}D")
        if X.shape[0] == 0:
            raise ValueError("No samples created - check your lookback period")
        if X.shape[1] != lookback or X.shape[2] != 2:
            raise ValueError(f"Invalid shape after preprocessing: {X.shape}. Expected (n, {lookback}, 2)")
            
        logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")
        return X, y, scaler
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise