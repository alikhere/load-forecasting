import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def get_gru_predictions(prediction_days):
    """
    Generate predictions using pre-trained GRU model
    Args:
        prediction_days: List of dates to predict (e.g., ['2025-03-25', '2025-03-26'])
    Returns:
        Dictionary containing metrics and predictions
    """
    # Directory paths
    actual_dir = './data/actual'
    predicted_dir = './data/simulated'
    model_dir = './grufiles'  # Directory containing model files
    
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
        # Load pre-trained model and scalers
        model = tf.keras.models.load_model(
            os.path.join(model_dir, 'gru_model.h5'),
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
        X_test = pd.read_csv(os.path.join(model_dir, 'X_test.csv')).values
        
        # Reshape test data
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Make predictions
        pred_scaled = model.predict(X_test_reshaped, verbose=0)
        pred_rescaled = scaler_y.inverse_transform(pred_scaled).flatten()
        
        # Combine actual data for all prediction days
        actual_dfs = []
        for day in prediction_days:
            actual_path = os.path.join(actual_dir, f"{day}.csv")
            actual_dfs.append(pd.read_csv(actual_path, parse_dates=['date_time']))
        actual_combined = pd.concat(actual_dfs)
        
        # Split predictions by day and calculate metrics
        start_idx = 0
        for day in prediction_days:
            day_df = actual_combined[actual_combined['date_time'].dt.date == pd.to_datetime(day).date()]
            day_actual = day_df['load'].values
            num_points = len(day_actual)
            day_predicted = pred_rescaled[start_idx:start_idx + num_points]
            start_idx += num_points
            
            # Calculate metrics
            mae = mean_absolute_error(day_actual, day_predicted)
            rmse = np.sqrt(mean_squared_error(day_actual, day_predicted))
            mape = np.mean(np.abs((day_actual - day_predicted) / day_actual)) * 100
            
            # Store results
            metrics['date'].append(day)
            metrics['actual'].append(day_actual.tolist())
            metrics['predicted'].append(day_predicted.tolist())
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
            metrics['MAPE'].append(mape)
            
            # Save daily predictions
            pred_df = pd.DataFrame({
                'date_time': day_df['date_time'],
                'load': day_predicted
            })
            os.makedirs(predicted_dir, exist_ok=True)
            pred_df.to_csv(os.path.join(predicted_dir, f"{day}_gru.csv"), index=False)
            
    except Exception as e:
        raise RuntimeError(f"GRU prediction failed: {str(e)}")
    
    return metrics