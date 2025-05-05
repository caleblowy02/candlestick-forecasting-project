import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Define the custom PositionalEncoding class
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.positional_encoding = self.get_positional_encoding(sequence_length, d_model)

    def get_positional_encoding(self, sequence_length, d_model):
        positions = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

# Load model
model = load_model("caleb_transformerr.h5", compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})

# Constants
STOCK_LIST = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "IBM"]
SEQ_LENGTH = 60
FORECAST_DAYS = 10

# Streamlit UI
st.title("📈 10-Day Stock Price Forecast")
selected_stock = st.selectbox("Select a stock", STOCK_LIST)

# Load historical data
data = yf.download(selected_stock, period="6mo")["Close"]

if len(data) < SEQ_LENGTH:
    st.error("Not enough data to generate prediction.")
    st.stop()

# Preprocess last 60 days
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
last_60 = scaled_data[-SEQ_LENGTH:]

# Forecast next 10 days
input_seq = last_60.copy()
predicted = []

for _ in range(FORECAST_DAYS):
    model_input = input_seq[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    pred_scaled = model.predict(model_input, verbose=0)
    predicted.append(pred_scaled[0, 0])
    input_seq = np.append(input_seq, pred_scaled)[-SEQ_LENGTH:]

# Inverse transform
predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
forecast_prices = np.insert(predicted_prices, 0, data.iloc[-1])
forecast_dates = pd.date_range(data.index[-1], periods=FORECAST_DAYS + 1, freq="B")

print("Last actual date:", data.index[-1])
print("First forecast date:", forecast_dates[0])
print("Time delta:", forecast_dates[0] - data.index[-1])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data[-SEQ_LENGTH:], label="Past 60 Days", color="steelblue")
plt.plot(forecast_dates, forecast_prices, label="Forecast (10 Days)", color="orange")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{selected_stock} Closing Price Forecast")
plt.legend()
st.pyplot(plt)

