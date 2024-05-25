import numpy as np
import pandas as pd
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.datasets import load_airline
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data_2023_2024 = read_csv('usdtry23-24.csv', usecols=[1], engine='python')
data_2023 = data_2023_2024[:364]
data_2024 = data_2023_2024[365:]

# Convert data to float and normalize
y = data_2023["Close"]

# Split the data into training and test sets
y_train, y_test = temporal_train_test_split(y, test_size=37)

# Define the forecasting horizon
fh = ForecastingHorizon(y_test.index, is_relative=False)

# Define a RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000)

# Use sktime's make_reduction to wrap the regressor for time series forecasting
forecaster = make_reduction(regressor, strategy="recursive", window_length=12)

# Fit the model on the training data
forecaster.fit(y_train)

# Make predictions
y_pred = forecaster.predict(fh)

# Evaluate the predictions
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape:.4f}')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: %.2f' % rmse)

# Plot the results
plt.plot(y_test.index, y_test, label='Actual Test Prices', color='brown')
plt.plot(y_pred.index, y_pred, label='Test Predicted 2023 Price', color='green')
plt.title('USD/TRY Daily Price')
plt.xlabel('Days')
plt.ylabel('Price (TRY)')
plt.legend()
plt.show()