from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)

def build_lstm_model(input_shape):
    """
    Build the LSTM model with improved architecture.
    """
    try:
        logger.info(f"Building LSTM model with input shape {input_shape}")
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape,
                      kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=64, return_sequences=False,
                      kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        logger.info("Model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise