
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta

st.set_page_config(page_title="Smart Money LSTM Forecast", layout="wide")
st.title("Smart Money AI LSTM Forecast Viewer")

ticker = st.text_input("Enter stock or crypto symbol (e.g. AAPL, BTC-USD)", "AAPL")

if ticker:
    df = yf.download(ticker, start='2021-01-01', end='2024-12-31')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    # Add features
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    df.dropna(inplace=True)

    # Scale and prepare data
    features = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'OBV', 'SMA_50', 'SMA_200']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    sequence_length = 60
    x = []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
    x = np.array(x)

    # Define and load model structure
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model in session (for demo; in production use a pre-trained model)
    y = scaled_data[sequence_length:, 0]
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    predictions = model.predict(x)
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['Close']])
    predicted_prices = close_scaler.inverse_transform(predictions)
    actual_prices = close_scaler.inverse_transform(y.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        'Date': df.index[-len(predicted_prices):],
        'Actual Price': actual_prices.flatten(),
        'Predicted Price': predicted_prices.flatten()
    })

    st.line_chart(forecast_df.set_index('Date'))

    st.dataframe(forecast_df.tail(10), use_container_width=True)
