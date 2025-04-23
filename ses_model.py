import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Settings
prediction_dir = './data/simulated'
actual_dir = './data/actual'

def initialize_weights(n_days=5, decay=0.5):
    weights = np.exp(-decay * np.arange(n_days))
    return weights / np.sum(weights)

def calculate_ses(data, weights):
    blocks_per_day = 96
    num_days = len(data) // blocks_per_day
    result = []
    for block in range(blocks_per_day):
        block_values = [
            data['load'].iloc[block + (num_days - i - 2) * blocks_per_day]
            for i in range(len(weights))
        ]
        ses_value = np.sum(np.array(block_values) * weights)
        result.append(ses_value)
    return result

def get_ses_predictions(prediction_days, decay=0.5):
    metrics_dict = {
        'date': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': [],
        'actual': [],
        'predicted': [],
        'weights': []
    }

    weights = initialize_weights(n_days=5, decay=decay)

    for i, target_day in enumerate(prediction_days):
        train_days = [f"2025-03-{d:02d}" for d in range(20 + i, 25 + i)]
        dfs = []

        for d in train_days:
            pred_path = os.path.join(prediction_dir, f"predicted{d}.csv")
            actual_path = os.path.join(actual_dir, f"{d}.csv")

            if os.path.exists(pred_path):
                df = pd.read_csv(pred_path)
            elif os.path.exists(actual_path):
                df = pd.read_csv(actual_path)
            else:
                raise FileNotFoundError(f"Missing data for {d}")
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['date_time'] = pd.to_datetime(combined_df['date_time'])

        # Predict
        predicted = calculate_ses(combined_df, weights)
        time_labels = pd.date_range(start=f'{target_day} 00:00', periods=96, freq='15min')
        pred_df = pd.DataFrame({'date_time': time_labels, 'load': predicted})

        # Save predictions
        save_path = os.path.join(prediction_dir, f"predicted{target_day}.csv")
        pred_df.to_csv(save_path, index=False)

        # Load actual
        actual_file = os.path.join(actual_dir, f"{target_day}.csv")
        actual_df = pd.read_csv(actual_file)
        actual = actual_df['load'].values

        # Metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Append
        metrics_dict['date'].append(target_day)
        metrics_dict['MAE'].append(mae)
        metrics_dict['RMSE'].append(rmse)
        metrics_dict['MAPE'].append(mape)
        metrics_dict['actual'].append(actual)
        metrics_dict['predicted'].append(predicted)
        metrics_dict['weights'].append(weights)

    return metrics_dict
