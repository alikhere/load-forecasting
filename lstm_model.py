# lstm_model.py

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_lstm_predictions(prediction_days):
    """
    Generate predictions using pre-trained LSTM model.
    Args:
        prediction_days: List of dates to predict (e.g., ['2025-03-25', ...])
    Returns:
        Dictionary containing metrics and predictions for each day
    """
    actual_dir = './data/actual'
    model_dir = './lstmfiles'
    predicted_dir = './data/simulated'
    os.makedirs(predicted_dir, exist_ok=True)

    # Initialize metrics storage
    metrics = {
        'date': [],
        'actual': [],
        'predicted': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': []
    }

    try:
        # Load model and scaler
        model = load_model(os.path.join(model_dir, 'lstm_model_cleaned.h5'))
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

        # Load scaled features
        val_X_scaled = pd.read_csv(
            os.path.join(model_dir, 'val_X_scaled.csv'),
            index_col='date_time',
            parse_dates=True
        )

        # Reshape for LSTM input
        val_X_reshaped = val_X_scaled.values.reshape((val_X_scaled.shape[0], 1, val_X_scaled.shape[1]))
        predictions_scaled = model.predict(val_X_reshaped, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled)

        predictions_df = pd.DataFrame(predictions, columns=['load'], index=val_X_scaled.index)

        for day in prediction_days:
            day_predictions = predictions_df[predictions_df.index.date == pd.to_datetime(day).date()]

            actual_file = os.path.join(actual_dir, f"{day}.csv")
            actual_df = pd.read_csv(actual_file, parse_dates=['date_time'], index_col='date_time')

            min_len = min(len(actual_df), len(day_predictions))
            actual = actual_df['load'].values[:min_len]
            predicted = day_predictions['load'].values[:min_len]

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            metrics['date'].append(day)
            metrics['actual'].append(actual.tolist())
            metrics['predicted'].append(predicted.tolist())
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
            metrics['MAPE'].append(mape)

            # Save daily predictions
            day_predictions.reset_index().to_csv(
                os.path.join(predicted_dir, f"{day}_lstm.csv"), index=False
            )

    except Exception as e:
        raise RuntimeError(f"LSTM prediction failed: {str(e)}")

    return metrics
