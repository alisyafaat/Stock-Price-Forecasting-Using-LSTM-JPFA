# Stock Price Forecasting Using LSTM – JPFA

## Overview

This project aims to develop a **Long Short-Term Memory (LSTM)** model for forecasting the stock price of PT Japfa Comfeed Indonesia Tbk (JPFA) using a *time series forecasting* approach. The LSTM model is designed to learn historical stock price patterns and generate predictions for subsequent periods.

This project focuses not only on evaluating model performance against actual data but also on conducting **out-of-sample forecasting**, which involves predicting stock prices beyond the last available actual observation. Therefore, the final results include a comparison between actual and predicted values on the testing data, as well as projected stock prices for future periods.

## Objectives

1. Develop an LSTM model to forecast JPFA stock prices based on historical data.
2. Compare model performance using different combinations of *LSTM Units* and *Batch Size*.
3. Evaluate prediction accuracy using Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).
4. Identify the hyperparameter combination that produces the best forecasting performance.
5. Perform *out-of-sample forecasting* to project stock prices beyond the available historical data.
6. Visualize actual stock prices, model predictions, and future price projections.

## Dataset

- **Stock:** PT Japfa Comfeed Indonesia Tbk (JPFA)
- **Stock Code:** JPFA
- **Data Type:** Historical Stock Price
- **Target Variable:** Adjusted Closing Price
- **Data Frequency:** Depending on the dataset used
- **Data Source:** Historical stock market data

## Methodology

The project consists of the following stages:

1. **Data Collection:** Collect historical stock price data for JPFA.
2. **Data Preprocessing:** Perform data inspection, handle missing values, and normalize the data using Min-Max Scaler.
3. **Sequence Preparation:** Transform time series data into sequential input patterns suitable for the LSTM model.
4. **Model Development:** Build the LSTM architecture and conduct experiments using different hyperparameter combinations.
5. **Model Evaluation:** Evaluate prediction performance using RMSE and MAPE.
6. **Best Model Selection:** Select the model with the best performance based on the evaluation results.
7. **Out-of-Sample Forecasting:** Use the selected model to predict stock prices beyond the last available actual observation.
8. **Visualization:** Present graphical comparisons of actual data, predicted values, and future stock price projections.

## Hyperparameter Experiment

Experiments are conducted using two variations of *LSTM Units* and two variations of *Batch Size* to investigate how different hyperparameter configurations affect model performance.

| Experiment | LSTM Units | Batch Size |
|:---:|:---:|:---:|
| Model 1 | 100 | 8 |
| Model 2 | 100 | 16 |
| Model 3 | 200 | 8 |
| Model 4 | 200 | 16 |

## Model Evaluation

Model performance is evaluated using two primary metrics:

- **Root Mean Squared Error (RMSE):** Measures the magnitude of prediction errors in stock price units. A lower RMSE indicates better model performance.
- **Mean Absolute Percentage Error (MAPE):** Measures the average percentage error between actual and predicted values. A lower MAPE indicates higher prediction accuracy.

### Evaluation Results

The evaluation results from all four hyperparameter combinations are compared to identify the model configuration with the lowest prediction errors.

| LSTM Units | Batch Size | RMSE | MAPE (%) |
|:---:|:---:|:---:|:---:|
| 100 | 8 | - | - |
| 100 | 16 | - | - |
| 200 | 8 | - | - |
| 200 | 16 | - | - |

### Evaluation Visualization

## Model Evaluation

Model performance is evaluated using two primary metrics:

- **Root Mean Squared Error (RMSE):** Measures the magnitude of prediction errors in stock price units. A lower RMSE indicates better model performance.
- **Mean Absolute Percentage Error (MAPE):** Measures the average percentage error between actual and predicted values. A lower MAPE indicates higher prediction accuracy.

### Evaluation Results

The evaluation is conducted using four different hyperparameter configurations, with each model consisting of four LSTM layers. The results are presented in the following table:

| Units | Batch Size | Layer | RMSE | MAPE (%) |
|:---:|:---:|:---:|---:|---:|
| 100 | 8 | 4 | 69.47248 | 4.475207 |
| 100 | 16 | 4 | 79.92947 | 4.899045 |
| 200 | 8 | 4 | **63.87585** | **4.302142** |
| 200 | 16 | 4 | 74.11518 | 4.555552 |

## Best Model Selection

Based on the evaluation results, the configuration of **200 LSTM Units and Batch Size 8** is selected as the best-performing model among the four tested configurations. This configuration achieves the most optimal forecasting performance based on the comparison of RMSE and MAPE values on the testing data.

**Best Model Configuration:**

| Parameter | Selected Value |
|:---:|:---:|
| LSTM Units | **200** |
| Batch Size | **8** |

The selected model is subsequently used to perform *out-of-sample forecasting*, generating JPFA stock price projections beyond the last available actual observation.

## Forecasting Results

After selecting the best-performing model, the forecasting process is carried out to project JPFA stock prices for future periods. The results include visualizations of historical stock prices, predictions on the testing data, and projected stock prices beyond the available actual data.

The following visualization presents the forecasting results generated using the LSTM model with the selected optimal hyperparameter configuration.

### Stock Price Forecasting Visualization



![JPFA Stock Price Forecasting](images/ForecastJPFA.JK.png.png)

*Figure: Actual Stock Prices, Testing Predictions, and Out-of-Sample Forecasting of JPFA.*

## Conclusion

Based on the experiments conducted, the combination of 200 LSTM Units and Batch Size 8 achieves the best performance among the four tested configurations. The selected model is then used to perform *out-of-sample forecasting*, generating projections of JPFA stock prices beyond the last available actual observation.

This project demonstrates the application of the Long Short-Term Memory (LSTM) method in time series analysis, covering data preprocessing, hyperparameter experimentation, model performance evaluation, best model selection, and future stock price forecasting.

## Disclaimer

This project is developed for educational purposes, data exploration, and the practical application of machine learning techniques in time series analysis. Forecasting results represent estimates based on historical patterns and should not be considered investment recommendations or guarantees of future stock price movements.
