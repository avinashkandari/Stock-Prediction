import numpy as np
import logging

logger = logging.getLogger(__name__)

def predict_next_10_days(model, last_sequence, scaler):
    """
    Predict stock prices for the next 10 days with error handling.
    """
    try:
        logger.info("Predicting next 10 days...")
        predictions = []
        current_sequence = last_sequence.copy()
        
        if current_sequence.shape != (60, 2):
            raise ValueError(f"Invalid sequence shape: {current_sequence.shape}. Expected (60, 2)")
        
        for _ in range(10):
            # Reshape the sequence to match LSTM input shape: (1, 60, 2)
            current_sequence_reshaped = current_sequence.reshape(1, 60, 2)
            
            # Predict the next day
            next_prediction = model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_prediction[0, 0])
            
            # Update the sequence with the new prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = next_prediction[0, 0]  # Update only the stock price
        
        # Inverse transform to get actual prices
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise