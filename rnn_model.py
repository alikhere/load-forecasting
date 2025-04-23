import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_rnn_predictions(prediction_days):
    """
    Generate predictions using pre-trained RNN model with WMA hybrid approach
    Args:
        prediction_days: List of dates to predict (e.g., ['2025-03-25', '2025-03-26'])
    Returns:
        Dictionary containing metrics and predictions
    """
    # Directory paths
    actual_dir = './data/actual'
    simulated_dir = './data/simulated '
    model_dir = './rnnfiles'
    
    # Initialize metrics dictionary
    metrics = {
        'date': [],
        'actual': [],
        'predicted': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': []
    }

    try:
        # Load pre-trained assets
        model = load_model(
            os.path.join(model_dir, 'rnn_model.h5'),
            custom_objects={"mse": "mean_squared_error"}
        )
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
        X_test = np.load(os.path.join(model_dir, 'X_test_reshaped.npy'))
        timestamps_df = pd.read_csv(os.path.join(model_dir, 'test_timestamps.csv'))
        
        # Make predictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Process timestamps (keep last 72 rows to match prediction length)
        timestamps_df = timestamps_df.iloc[24:].reset_index(drop=True)
        
        # Create base prediction DataFrame
        rnn_pred_df = pd.DataFrame({
            'date_time': timestamps_df['date_time'],
            'load': y_pred
        })
        
        # Load WMA predictions for first 24 blocks (March 25)
        wma_df = pd.read_csv(os.path.join(simulated_dir, '2025-03-25_sim.csv'))
        wma_first_24 = wma_df.iloc[:24]
        
        # Combine WMA first 24 with RNN predictions
        final_pred_df = pd.concat([wma_first_24, rnn_pred_df], ignore_index=True)
        
        # Process each prediction day
        for day in prediction_days:
            # Filter predictions for current day
            day_pred_df = final_pred_df[
                final_pred_df['date_time'].str.startswith(day)
            ].copy()
            
            # Load actual data
            actual_df = pd.read_csv(
                os.path.join(actual_dir, f"{day}.csv"),
                parse_dates=['date_time']
            )
            
            # Align lengths
            min_len = min(len(actual_df), len(day_pred_df))
            actual = actual_df['load'].values[:min_len]
            predicted = day_pred_df['load'].values[:min_len]
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Store results
            metrics['date'].append(day)
            metrics['actual'].append(actual.tolist())
            metrics['predicted'].append(predicted.tolist())
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
            metrics['MAPE'].append(mape)
            
            # Save daily predictions
            os.makedirs(simulated_dir, exist_ok=True)
            day_pred_df.to_csv(
                os.path.join(simulated_dir, f"{day}_rnn.csv"),
                index=False
            )
            
    except Exception as e:
        raise RuntimeError(f"RNN prediction failed: {str(e)}")
    
    return metrics