import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# Paths (consistent with your structure)
actual_path = './data/actual'
predicted_path = './data/simulated'
os.makedirs(predicted_path, exist_ok=True)

def get_wma_predictions(prediction_days):
    """Calculate WMA predictions and metrics for given prediction days"""
    metrics = {
        'date': [],
        'actual': [],
        'predicted': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': [],
        'weights': []
    }

    # Get training files (March 1-24)
    train_files = [os.path.join(actual_path, f"2025-03-{day:02d}.csv") for day in range(1, 25)]
    
    # Calculate min/max for normalization
    min_val, max_val = get_min_max(train_files)
    
    # Train model and get weights
    weights = train_wma_model(train_files, min_val, max_val)
    
    for day in prediction_days:
        # Predict for current day
        predicted = predict_day(day, weights, min_val, max_val)
        
        # Get actual values
        actual_file = os.path.join(actual_path, f"{day}.csv")
        actual_df = pd.read_csv(actual_file)
        actual = actual_df['load'].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Store results
        metrics['date'].append(day)
        metrics['actual'].append(actual.tolist())
        metrics['predicted'].append(predicted)  # FIXED: Removed .tolist()
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['MAPE'].append(mape)
        metrics['weights'].append(weights.copy())
        
        # Save prediction
        save_prediction(day, actual_df, predicted)
    
    return metrics

def train_wma_model(train_files, min_val, max_val):
    """Train WMA model using gradient descent"""
    weight = [0.2] * 5  # Initial weights
    epochs = 1000  # Reduced for Streamlit (original was 8000)
    lr = 1e-2
    
    # Load and normalize training data
    data_dict = {}
    for file in train_files:
        df = pd.read_csv(file)
        data_dict[file] = normalize(df['load'].values, min_val, max_val)
    
    # Gradient descent
    for epoch in range(epochs):
        grad = [0.0] * len(weight)
        total_blocks = (len(train_files) - len(weight)) * len(data_dict[train_files[0]])
        
        for i in range(len(weight), len(train_files)):
            for block in range(len(data_dict[train_files[0]])):
                forecast_val = sum(weight[k] * data_dict[train_files[i - k - 1]][block] 
                                 for k in range(len(weight)))
                error = forecast_val - data_dict[train_files[i]][block]
                
                for k in range(len(weight)):
                    grad[k] += (error * data_dict[train_files[i - k - 1]][block]) / total_blocks
        
        # Update weights
        for k in range(len(weight)):
            weight[k] -= lr * grad[k]
    
    return weight

def predict_day(day, weights, min_val, max_val):
    """Predict load for a specific day using WMA"""
    # Get input files (previous 5 days)
    input_files = []
    for i in range(5):
        date = datetime.strptime(day, "%Y-%m-%d") - timedelta(days=5 - i)
        input_file = os.path.join(actual_path, f"{date.date()}.csv")
        input_files.append(input_file)
    
    # Load and normalize input data
    input_data = []
    for file in input_files:
        df = pd.read_csv(file)
        input_data.append(normalize(df['load'].values, min_val, max_val))
    
    # Make prediction
    prediction_norm = []
    for block in range(len(input_data[0])):
        pred = sum(weights[i] * input_data[4 - i][block] for i in range(5))
        prediction_norm.append(pred)
    
    # Denormalize and return
    return denormalize(prediction_norm, min_val, max_val)

def save_prediction(day, actual_df, predicted):
    """Save prediction to CSV"""
    output_df = pd.DataFrame({
        'date_time': actual_df['date_time'],
        'load': predicted
    })
    output_df.to_csv(os.path.join(predicted_path, f"{day}.csv"), index=False)

def get_min_max(files):
    """Get min and max load values from files"""
    all_loads = []
    for file in files:
        df = pd.read_csv(file)
        all_loads.extend(df['load'].tolist())
    return min(all_loads), max(all_loads)

def normalize(values, min_val, max_val):
    """Normalize values to [0,1] range"""
    return [(v - min_val) / (max_val - min_val) for v in values]

def denormalize(values, min_val, max_val):
    """Denormalize values from [0,1] range"""
    return [v * (max_val - min_val) + min_val for v in values]
