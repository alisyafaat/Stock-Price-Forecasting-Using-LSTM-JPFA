import math
import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
import warnings
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
# SET RANDOM SEED
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# LOAD DATA
csv_path = r"C:\\Users\\syafa\\Downloads\\Data\\JPFA.JK.csv"
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
weekly = df[['AdjClosePrice']]
n_steps = 30
test_weeks = 52
epochs = 100

training_set = weekly.iloc[:-test_weeks].values
testing_set  = weekly.iloc[-test_weeks:].values
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(training_set)

X_train, y_train = [], []

for i in range(n_steps, len(training_scaled)):
    X_train.append(training_scaled[i-n_steps:i, 0])
    y_train.append(training_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

dataset_train = weekly.iloc[:-test_weeks]
dataset_test  = weekly.iloc[-test_weeks:]

dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
inputs = dataset_total[
    len(dataset_total) - len(dataset_test) - n_steps:
].values

inputs = sc.transform(inputs)
X_test, y_test = [], []

for i in range(n_steps, n_steps + len(dataset_test)):
    X_test.append(inputs[i-n_steps:i, 0])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

def build_and_evaluate(units, batch_size):

    model = Sequential()
    model.add(LSTM(units, return_sequences=True,
                   input_shape=(n_steps, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred = sc.inverse_transform(y_pred)

    y_true = dataset_test.values.flatten()

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(
        np.abs((y_true - y_pred.flatten()) / y_true)
    ) * 100

    return rmse, mape

param_grid = [
    (100, 8),
    (100, 16),
    (200, 8),
    (200, 16)
]

results = []

for units, batch in param_grid:
    rmse, mape = build_and_evaluate(units, batch)

    results.append({
        "Units": units,
        "Batch Size": batch,
        "Layer": 4,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

result_df = pd.DataFrame(results)
result_df

print(result_df)


