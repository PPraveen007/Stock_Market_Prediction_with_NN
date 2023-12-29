# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
import numpy as np
import pandas as pd 
import tensorflow as tf 
import keras 
from keras.layers import LSTM, Dense 
from keras.models import Sequential
from keras import optimizers
from prophet.forecaster import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


def predict_value(params):
    lr = 1e-3
    rmodel = Sequential()
    rmodel.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='relu'))
    rmodel.add(LSTM(40, activation='relu', return_sequences=True))
    rmodel.add(LSTM(35, activation='relu', return_sequences=True))
    rmodel.add(Dense(1))
    rmodel.compile(optimizer=tf.optimizers.Adam(lr=lr), loss='mean_squared_error')
    rmodel.summary()

    rmodel.load_weights('MV3-LSTM-Split3.h5')
    predicted_value = rmodel.predict([[params]])
    
    return predicted_value[0][0][0] 

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
print(df_train.head())
df_train['ds'] = df_train['ds'].apply(lambda x: x.replace(tzinfo=None))

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)