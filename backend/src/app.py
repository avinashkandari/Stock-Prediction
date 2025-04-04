from flask import Flask, request, jsonify
from flask_cors import CORS
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import fetch_stock_data, fetch_news_articles, analyze_sentiment
from preprocessor import preprocess_data
from model import build_lstm_model
from predictor import predict_next_10_days
import threading
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)
CORS(app)

# Set to Indian timezone
IST = pytz.timezone('Asia/Kolkata')
app.config['TIMEZONE'] = IST

# Global variables
trained_models = {}
model_lock = threading.Lock()
training_status = {}  # Tracks training progress per ticker

EODHD_API_KEY = "67c3793e67ebc4.02813409"

@app.route('/api/status/<ticker>', methods=['GET'])
def get_status(ticker):
    """Check training status for a ticker"""
    with model_lock:
        if ticker in trained_models:
            return jsonify({'status': 'ready'})
        if ticker in training_status:
            return jsonify({
                'status': 'training',
                'progress': training_status[ticker].get('progress', 0)
            })
    return jsonify({'status': 'not_started'}), 404

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict(ticker):
    """Get predictions for a ticker"""
    with model_lock:
        if ticker in trained_models:
            model_data = trained_models[ticker]
            return jsonify({
                'status': 'ready',
                'historical': {
                    'dates': model_data['dates'],
                    'actual': model_data['actual_prices'],
                    'predicted': model_data['predicted_prices']
                },
                'forecast': {
                    'dates': model_data['future_dates'],
                    'predicted': model_data['next_10_days']
                },
                'metrics': model_data['metrics']
            })
        
        if ticker in training_status:
            return jsonify({
                'status': 'training',
                'message': f'Training {ticker} (Epoch {training_status[ticker].get("epoch", 0)}/10)',
                'progress': training_status[ticker].get('progress', 0)
            }), 202
        
        return jsonify({
            'status': 'not_started',
            'message': f'Model for {ticker} not trained yet'
        }), 404

@app.route('/api/train', methods=['POST'])
def train():
    """Start training for a ticker"""
    data = request.json
    ticker = data.get('ticker', '').upper()
    start_date = data.get('start_date', '2020-01-01')
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol required'}), 400
    
    with model_lock:
        if ticker in trained_models:
            return jsonify({'status': 'ready', 'message': 'Model already trained'}), 200
        if ticker in training_status:
            return jsonify({'status': 'training', 'message': 'Training already in progress'}), 200
        
        training_status[ticker] = {'status': 'training', 'progress': 0, 'epoch': 0}
    
    threading.Thread(
        target=train_model_async,
        args=(ticker, start_date, EODHD_API_KEY),
        daemon=True
    ).start()
    
    return jsonify({
        'status': 'training_started',
        'ticker': ticker,
        'message': f'Training started for {ticker}. Poll /api/predict/{ticker} for updates.'
    }), 202

def train_model_async(ticker, start_date, api_key):
    try:
        # Get current date in IST
        current_date = datetime.now(IST)
        end_date = current_date.strftime('%Y-%m-%d')
        
        # 1. Fetch stock data
        stock_prices, dates = fetch_stock_data(ticker, start_date, end_date, api_key)
        if stock_prices is None:
            raise ValueError(f"Failed to fetch stock data for {ticker}")
        
        # Convert dates to datetime objects in IST
        dates = [datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=IST) if isinstance(d, str) else d for d in dates]
        
        # 2. Fetch and analyze news sentiment
        articles = fetch_news_articles(api_key, ticker)
        sentiments = analyze_sentiment(articles) if articles else np.zeros(len(stock_prices))
        
        # 3. Preprocess data
        min_length = min(len(stock_prices), len(sentiments))
        stock_prices = stock_prices[:min_length]
        sentiments = sentiments[:min_length]
        dates = dates[:min_length]
        
        X, y, scaler = preprocess_data(stock_prices, sentiments)
        
        # 4. Train model
        model = build_lstm_model((X.shape[1], X.shape[2]))
        for epoch in range(10):
            model.fit(X, y, batch_size=32, epochs=1, verbose=0)
            with model_lock:
                training_status[ticker] = {
                    'status': 'training',
                    'progress': (epoch + 1) * 10,
                    'epoch': epoch + 1
                }
        
        # 5. Make predictions
        predicted_prices = scaler.inverse_transform(model.predict(X)).flatten()
        actual_prices = stock_prices[60:].flatten()
        historical_dates = dates[60:]
        
        # 6. Forecast next 10 days from TODAY
        last_sequence = np.hstack((X[-1, :, 0].reshape(-1, 1), sentiments[-60:].reshape(-1, 1)))
        next_10_days = predict_next_10_days(model, last_sequence, scaler)
        
        # 7. Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)
        
        # 8. Prepare future dates starting from tomorrow
        future_dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 11)]
        
        # 9. Store results
        with model_lock:
            trained_models[ticker] = {
                'actual_prices': actual_prices.tolist(),
                'predicted_prices': predicted_prices.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in historical_dates],
                'next_10_days': next_10_days.tolist(),
                'future_dates': future_dates,
                'metrics': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'last_updated': current_date.isoformat(),
                    'prediction_date': current_date.strftime('%Y-%m-%d')
                }
            }
            training_status.pop(ticker, None)
        
        print(f"\n=== Successfully trained model for {ticker} ===")
        print(f"Prediction date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Forecast dates: {future_dates}")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
    except Exception as e:
        print(f"\n!!! Training failed for {ticker}: {str(e)}")
        with model_lock:
            training_status[ticker] = {'status': 'failed', 'error': str(e)}
    finally:
        print("Training process completed")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)