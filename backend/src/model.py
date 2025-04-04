from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

def build_lstm_model(input_shape):
    """
    Build the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, epochs=10, batch_size=32, validation_split=0.1):
    """
    Train the LSTM model.
    """
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)