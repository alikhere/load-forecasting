import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_sarima_predictions(prediction_days):
    """
    Generate predictions using SARIMA model
    Args:
        prediction_days: List of dates to predict (e.g., ['2025-03-25', '2025-03-26'])
    Returns:
        Dictionary containing metrics and predictions
    """
    # Directory paths
    actual_dir = './data/actual'
    model_dir = './arimafiles'
    predicted_dir = './data/simulated'
    os.makedirs(predicted_dir, exist_ok=True)
    
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
        # Load training and test data
        train_data = pd.read_csv(
            os.path.join(model_dir, 'train_data.csv'),
            parse_dates=['date_time'],
            index_col='date_time'
        )['load']  # Load just the load column as Series
        
        test_data = pd.read_csv(
            os.path.join(model_dir, 'test_data.csv'),
            parse_dates=['date_time'],
            index_col='date_time'
        )['load']

        # Load the pre-trained SARIMA model
        sarima_results = joblib.load(os.path.join(model_dir, 'sarima_model.pkl'))

        # Generate predictions for all test dates
        predictions = sarima_results.predict(
            start=len(train_data),
            end=len(train_data) + len(test_data) - 1,
            dynamic=False
        )

        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'date_time': test_data.index,
            'load': predictions.values
        })

        # Process each prediction day
        for day in prediction_days:
            # Filter predictions for current day
            day_predictions = predictions_df[
                predictions_df['date_time'].dt.date == pd.to_datetime(day).date()
            ]
            
            # Load actual data
            actual_df = pd.read_csv(
                os.path.join(actual_dir, f"{day}.csv"),
                parse_dates=['date_time']
            )
            
            # Align lengths
            min_len = min(len(actual_df), len(day_predictions))
            actual = actual_df['load'].values[:min_len]
            predicted = day_predictions['load'].values[:min_len]
            
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
            
            # Save daily predictions with SARIMA suffix
            day_predictions.to_csv(
                os.path.join(predicted_dir, f"{day}_sarima.csv"),
                index=False
            )
            
    except Exception as e:
        raise RuntimeError(f"SARIMA prediction failed: {str(e)}")
    
    return metrics
