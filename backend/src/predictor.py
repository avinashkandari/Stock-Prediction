import numpy as np

def predict_next_10_days(model, last_sequence, scaler):
    """
    Predict stock prices for the next 10 days.
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(10):
        # Reshape the sequence to match LSTM input shape: (1, 60, 2)
        current_sequence_reshaped = current_sequence.reshape(1, 60, 2)
        
        # Predict the next day
        next_prediction = model.predict(current_sequence_reshaped)
        predictions.append(next_prediction[0, 0])
        
        # Update the sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1, 0] = next_prediction[0, 0]  # Update only the stock price, not the sentiment score
    
    # Inverse transform to get actual prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()