import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
actual_path = './data/actual'
predicted_path = './data/simulated'
os.makedirs(predicted_path, exist_ok=True)

# Function to get SMA predictions and metrics
def get_sma_predictions(prediction_days):
    metrics = {
        'date': [],
        'actual': [],
        'predicted': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': []
    }

    for day in prediction_days:
        print(f"\n🔄 Predicting for: {day}")

        # Get training days: March 15 to current_day - 1
        start_train = pd.to_datetime("2025-03-15")
        current_day = pd.to_datetime(day)
        train_days = pd.date_range(start=start_train, end=current_day - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        train_dfs = []

        for train_day in train_days:
            file_path = os.path.join(actual_path, f"{train_day}.csv")
            df = pd.read_csv(file_path, parse_dates=['date_time'], index_col='date_time')
            df['time'] = df.index.time
            train_dfs.append(df)

        # Combine all training data
        train = pd.concat(train_dfs)
        train = train.sort_index()

        # Load actual test file (for metrics only)
        test_file = os.path.join(actual_path, f"{day}.csv")
        test = pd.read_csv(test_file, parse_dates=['date_time'], index_col='date_time')
        test = test.sort_index()
        test['time'] = test.index.time

        # Compute block-wise average
        blockwise_avg = train.groupby('time')['load'].mean()

        # Predict using SMA
        test['predicted_load'] = test['time'].map(blockwise_avg)

        # Save predicted values for next iteration
        predicted_df = pd.DataFrame({
            'date_time': test.index,
            'load': test['predicted_load']
        })
        predicted_df.to_csv(f"{predicted_path}/{day}.csv", index=False)

        # Evaluation
        mae = mean_absolute_error(test['load'], test['predicted_load'])
        rmse = np.sqrt(mean_squared_error(test['load'], test['predicted_load']))
        mape = (abs((test['load'] - test['predicted_load']) / test['load']).mean()) * 100

        # Save metrics and actual/predicted values for plotting
        metrics['date'].append(day)
        metrics['actual'].append(test['load'].values.tolist())
        metrics['predicted'].append(test['predicted_load'].values.tolist())
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['MAPE'].append(mape)

    return metrics
