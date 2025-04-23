import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from sma_model import get_sma_predictions
from wma_model import get_wma_predictions
from ses_model import get_ses_predictions  
from gru_model import get_gru_predictions
from rnn_model import get_rnn_predictions
from lstm_model import get_lstm_predictions
from arima_model import get_sarima_predictions

# Streamlit app configuration
st.set_page_config(layout="wide", page_title="Load Forecasting Dashboard")

# Title and sidebar
st.title("⚡ Load Forecasting Dashboard")
st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    [
        "Simple Moving Average (SMA)", 
        "Weighted Moving Average (WMA)", 
        "Simple Exponential Smoothing (SES)",
        "Gated Recurrent Unit (GRU)",
        "Recurrent Neural Network (RNN)",
        "Long Short-Term Memory (LSTM)",
        "Seasonal ARIMA (SARIMA)"
    ],
    index=0
)

# Prediction days
prediction_days = ['2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28', '2025-03-29']

# Paths
actual_path = './data/actual'
predicted_path = './data/simulated'

# Get predictions based on selected model
@st.cache_data
def get_predictions(model_name, days):
    if model_name == "Simple Moving Average (SMA)":
        return get_sma_predictions(days)
    elif model_name == "Weighted Moving Average (WMA)":
        return get_wma_predictions(days)
    elif model_name == "Simple Exponential Smoothing (SES)":
        return get_ses_predictions(days)
    elif model_name == "Gated Recurrent Unit (GRU)":
        return get_gru_predictions(days)
    elif model_name == "Recurrent Neural Network (RNN)":
        return get_rnn_predictions(days)
    elif model_name == "Long Short-Term Memory (LSTM)":
        return get_lstm_predictions(days)
    else:  # SARIMA
        return get_sarima_predictions(days)
# Load the selected model's predictions
try:
    metrics = get_predictions(selected_model, prediction_days)
except Exception as e:
    st.error(f"Error loading {selected_model}: {str(e)}")
    st.stop()

# =============================================
# MAIN DASHBOARD LAYOUT
# =============================================

# Create two main columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"{selected_model} Forecast Visualization")
    
    # Date selector for specific day analysis
    selected_date = st.selectbox("Select Date for Detailed View", prediction_days)
    selected_idx = prediction_days.index(selected_date)
    
    # Load actual data for plotting
    actual_file = os.path.join(actual_path, f"{selected_date}.csv")
    actual_df = pd.read_csv(actual_file, parse_dates=['date_time'])
    
    # Plot actual vs predicted
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    time_labels = pd.to_datetime(actual_df['date_time'])
    ax1.plot(time_labels, metrics['actual'][selected_idx], 
             label='Actual Load', linewidth=2, color='#1f77b4')
    ax1.plot(time_labels, metrics['predicted'][selected_idx], 
             label='Predicted Load', linestyle='--', linewidth=2, color='#ff7f0e')
    ax1.set_title(f"Actual vs Predicted Load - {selected_date} ({selected_model})", pad=20)
    ax1.set_xlabel("Time", labelpad=10)
    ax1.set_ylabel("Load (MW)", labelpad=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.header("Model Performance")
    
    # Display weights if WMA is selected
    if selected_model == "Weighted Moving Average (WMA)" and 'weights' in metrics:
        st.subheader("Model Weights")
        weights_df = pd.DataFrame({
            'Window': [f"t-{i}" for i in range(len(metrics['weights'][0]))][::-1],
            'Weight': metrics['weights'][0]
        })
        st.dataframe(weights_df.style.format({'Weight': '{:.4f}'}))

    # Metrics for selected date
    st.subheader(f"Metrics for {selected_date}")
    selected_metrics = {
        'Metric': ['MAE (MW)', 'RMSE (MW)', 'MAPE (%)'],
        'Value': [
            metrics['MAE'][selected_idx],
            metrics['RMSE'][selected_idx],
            metrics['MAPE'][selected_idx]
        ]
    }
    st.table(pd.DataFrame(selected_metrics).style.format({'Value': '{:.2f}'}))

    # All metrics summary
    st.subheader("All Dates Summary")
    summary_df = pd.DataFrame({
        'Date': metrics['date'],
        'MAE (MW)': metrics['MAE'],
        'RMSE (MW)': metrics['RMSE'],
        'MAPE (%)': metrics['MAPE']
    })
    st.dataframe(summary_df.style.format({
        'MAE (MW)': '{:.2f}', 
        'RMSE (MW)': '{:.2f}', 
        'MAPE (%)': '{:.2f}'
    }))

# =============================================
# ERROR ANALYSIS SECTION
# =============================================
st.header("📊 Error Analysis")

# Plot MAE and RMSE trends
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(metrics['date'], metrics['MAE'], marker='o', label='MAE', color='#2ca02c')
ax2.plot(metrics['date'], metrics['RMSE'], marker='s', label='RMSE', color='#d62728')
ax2.set_title(f"Error Trends Over Time ({selected_model})", pad=15)
ax2.set_xlabel("Prediction Date", labelpad=10)
ax2.set_ylabel("Error Value", labelpad=10)
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)

# Plot MAPE separately
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(metrics['date'], metrics['MAPE'], marker='^', label='MAPE', color='#9467bd')
ax3.set_title(f"MAPE Over Time ({selected_model})", pad=15)
ax3.set_xlabel("Prediction Date", labelpad=10)
ax3.set_ylabel("MAPE (%)", labelpad=10)
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig3)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("### 💡 Tips")
st.info("""
- Select different models from the sidebar to compare their performance
- Click on any date to see detailed forecasts and metrics
- Hover over charts to see exact values
""")
