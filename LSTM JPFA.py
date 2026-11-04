import math
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import warnings
import os
import random
import tensorflow as tf

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4f}'.format

# 1. Load data mingguan dari CSV
csv_path = r"C:\\Users\\syafa\\Downloads\\Data\\JPFA.JK.csv"  # GANTI PATH
df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['AdjClosePrice'] = (
    df['AdjClosePrice']
    .astype(str)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

df = df.sort_values('Date').set_index('Date')

print(df.head())
print(df.tail())
print(df.index.dtype)

# Validasi kolom
if 'AdjClosePrice' not in df.columns:
    raise ValueError("Kolom AdjClosePrice tidak ditemukan")

weekly = df[['AdjClosePrice']]

print(f"Total minggu data: {len(weekly)}")
print(weekly.head())


# PLOT HARGA AKTUAL (PRE-LSTM)

plt.figure(figsize=(10, 5))

plt.plot(
    weekly.index,
    weekly['AdjClosePrice'],
    color='#00FFFF',
    linewidth=2,
    label='Actual Price (Weekly)'
)

plt.title('JPFA.JK Weekly Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Price (IDR)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    r"C:\\Users\\syafa\\Downloads\\plot\\JPFAA_Actual1.png",
    dpi=300
)

plt.show()

# 2. Parameter utama
n_steps = 30           # window LSTM (30 minggu)
test_weeks = 52        # 1 tahun terakhir sebagai test
forecast_months = 5    # horizon forecast (bulan)

# 3. Train / Test split
training_set = weekly.iloc[:-test_weeks].values
testing_set  = weekly.iloc[-test_weeks:].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
dataset_train_scaled = sc.fit_transform(training_set)

# Creating a data structure with 30 time-steps and 1 output
X_train = []
y_train = []

for i in range(n_steps, len(dataset_train_scaled)):
    X_train.append(dataset_train_scaled[i-n_steps:i, 0])
    y_train.append(dataset_train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model = Sequential()

#Adding the first LSTM layer and some Dropout regularJPFAion
model.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularJPFAion
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularJPFAion
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularJPFAion
model.add(LSTM(units = 200))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 8)

# Getting the predicted stock price of 2024
dataset_train = weekly.iloc[:-test_weeks]
dataset_test = weekly.iloc[-test_weeks:]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - n_steps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
y_test = []
for i in range(n_steps, n_steps + len(dataset_test)):
    X_test.append(inputs[i-n_steps:i, 0])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results (TESTING DATA)
plt.figure(figsize=(10, 5))
plt.plot(dataset_test.values, color='#00FFFF', label='Real JPFA Price', linewidth=2)
plt.plot(predicted_stock_price, color='#00FF00', label='Predicted JPFA Price', linewidth=2)
plt.xticks(np.arange(0, 100, 50))
plt.title('JPFA Price Prediction (Testing Data)')
plt.xlabel('Time')
plt.ylabel('Price (IDR)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    r"C:\\Users\\syafa\\Downloads\\plot\\TestingJPFA1.JK.png",
    dpi=300
)
plt.show()

predicted_stock_price
df1 = pd.DataFrame(data=predicted_stock_price,columns=['Predicted price'])
df2 = pd.DataFrame(data=dataset_test.values,columns=['Actual price'])

result2 = pd.concat([df1, df2], axis=1, join="inner")
result2

len(dataset_test)

# Jumlah time-step (konsisten dengan training)
n_steps = 30

# Mengambil n_steps terakhir dari data testing
temp_input = dataset_test.values[-n_steps:]
temp_input = temp_input.reshape(-1).tolist()

# EVALUATION METRICS (TESTING DATA)
actual_price    = dataset_test.values.flatten()
predicted_price = predicted_stock_price.flatten()

# RMSE
rmse = math.sqrt(mean_squared_error(actual_price, predicted_price))

# MAPE
mape = np.mean(
    np.abs((actual_price - predicted_price) / actual_price)
) * 100

print("RMSE :", rmse)
print("MAPE :", mape, "%")

# DATE-BASED MULTI STEP FORECAST
last_date = weekly.index[-1]
end_forecast = last_date + pd.DateOffset(months=forecast_months)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=7),
    end=end_forecast,
    freq='W-FRI'
)

forecast_steps = len(future_dates)

lst_output = []
temp_input = inputs[-n_steps:].flatten().tolist()

i = 0
while i < forecast_steps:

    x_input = np.array(temp_input)
    x_input = x_input.reshape((1, n_steps, 1))

    yhat = model.predict(x_input, verbose=0)

    lst_output.append(yhat[0, 0])

    temp_input.pop(0)
    temp_input.append(yhat[0, 0])

    i += 1

print(lst_output)

real_predicted_stock_price = sc.inverse_transform(
    np.array(lst_output).reshape(-1, 1)
)

print(real_predicted_stock_price)

pred_price = pd.DataFrame(data=real_predicted_stock_price,columns=['Predicted price'])

pred_price.to_excel('C:\\Users\\syafa\\Downloads\\Forcasting\\JPFA.F1.xlsx', index = False)

# FINAL PLOT: DATE-BASED (CORRECT)
# Actual prices (entire historical data)
actual_prices = weekly.iloc[:, 0].values
actual_dates  = weekly.index

# Forecast prices (after last date)
forecast_prices = real_predicted_stock_price.flatten()
forecast_dates  = future_dates

plt.figure(figsize=(10, 5))

plt.plot(actual_dates, actual_prices,
         label='Actual Price (Historical)',
         color='#00FFFF', linewidth=2)

plt.plot(forecast_dates, forecast_prices,
         label='Forecast Price (Next 5 Months)',
         color='#00FF00', linewidth=2)

plt.title('JPFA Price Forecasting using LSTM')
plt.xlabel('Time')
plt.ylabel('Price (IDR)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(r"C:\\Users\\syafa\\Downloads\\plot\\ForecastJPFA.JK1.png",
    dpi=300
)
plt.show()
