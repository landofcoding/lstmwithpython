# LSTM for USD/TRY daily price prediction with window regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential

# Function to create a dataset with windowed sequences
def windowed_sequences(ds, past_period=1):
    sequenceX, sequenceY = [], []
    for i in range(len(ds) - past_period - 1):
        a = ds[i:(i + past_period), 0]  # Extract past_period values
        sequenceX.append(a)
        sequenceY.append(ds[i + past_period, 0])  # next day prices
    return np.array(sequenceX), np.array(sequenceY)


# to get output given the same input
np.random.seed(7)

# Load USD/TRY data for 2023 and 2024
data_2023_2024 = read_csv('usdtry23-24.csv', usecols=[1], engine='python')

# Convert data to float and normalize
price_data_2023_2024 = data_2023_2024.values.astype('float32')

plt.plot(price_data_2023_2024, color='brown', label='Historical Price')
plt.title('USD/TRY Daily Prices')
plt.xlabel('Days')
plt.ylabel('Price (TRY)')
plt.legend()
plt.show()

minMaxScaler = MinMaxScaler(feature_range=(0, 1))
normalized_data_2023_2024 = minMaxScaler.fit_transform(price_data_2023_2024)
normalized_data_2023 = normalized_data_2023_2024[:364]
normalized_data_2024 = normalized_data_2023_2024[365:]

# Split into training dataset and testing
trainDataSize = int(len(normalized_data_2023) * 0.80)
testDataSize = len(normalized_data_2023) - trainDataSize
trainDataset, testDataset = normalized_data_2023[0:trainDataSize, :], normalized_data_2023[trainDataSize:len(normalized_data_2023), :]

# Define prediction data (all of 2024 data)
predictDataset24 = normalized_data_2024

# Create windowed datasets for training, testing, and prediction
past_period = 5
trainDatasetX, trainDatasetY = windowed_sequences(trainDataset, past_period)
testDatasetX, testDatasetY = windowed_sequences(testDataset, past_period)
predictDatasetX, predictDatasetY = windowed_sequences(predictDataset24, past_period)

# Reshape data for LSTM input
trainDatasetX = np.reshape(trainDatasetX, (trainDatasetX.shape[0], 1, trainDatasetX.shape[1]))
testDatasetX = np.reshape(testDatasetX, (testDatasetX.shape[0], 1, testDatasetX.shape[1]))
predictDatasetX = np.reshape(predictDatasetX, (predictDatasetX.shape[0], 1, predictDatasetX.shape[1]))

# Build LSTM model
steps_out=1
lstmModel = Sequential()
lstmModel.add(LSTM(100, return_sequences=False, input_shape=(1, past_period)))  # Single LSTM layer with 100 units
lstmModel.add(Dense(steps_out))
lstmModel.compile(loss='mean_squared_error', optimizer='adam')
lstmModel.fit(trainDatasetX, trainDatasetY, epochs=100, batch_size=1, verbose=2)

# Make predictions on training, testing, and 2024 data
trainPredict = lstmModel.predict(trainDatasetX)
testPredict = lstmModel.predict(testDatasetX)
predict24 = lstmModel.predict(predictDatasetX)

# Invert predictions back to original scale
trainPredict = minMaxScaler.inverse_transform(trainPredict)
testPredict = minMaxScaler.inverse_transform(testPredict)
predict24 = minMaxScaler.inverse_transform(predict24)

trainDatasetY = minMaxScaler.inverse_transform([trainDatasetY])
testDatasetY = minMaxScaler.inverse_transform([testDatasetY])
predictDatasetY = minMaxScaler.inverse_transform([predictDatasetY])


#Calculate rmse for train, test, and 2024 prediction
trainDatasetScore =np.sqrt(mean_squared_error(trainDatasetY[0], trainPredict[:,0]))
print('Train dataset RMSE: %.2f' % (trainDatasetScore))
testDatasetScore = np.sqrt(mean_squared_error(testDatasetY[0], testPredict[:,0]))
print('Test dataset RMSE: %.2f' % testDatasetScore)
predictDatasetScore = np.sqrt(mean_squared_error(predictDatasetY[0], predict24[:,0]))
print('Predict 2024 RMSE: %.2f' % predictDatasetScore)

# place train dataset for plotting
train_pred_plot = np.empty_like(normalized_data_2023_2024)
train_pred_plot[:, :] = np.nan
train_pred_plot[past_period:len(trainPredict) + past_period, :] = trainPredict

# place test dataset for plotting
test_pred_plot = np.empty_like(normalized_data_2023)
test_pred_plot[:, :] = np.nan
test_pred_plot[len(trainPredict) + (past_period * 2) + 1:len(normalized_data_2023) - 1, :] = testPredict

# Shift 24 predictions for plotting
pred24_plot = np.empty_like(normalized_data_2023_2024)
pred24_plot[:, :] = np.nan
pred24_plot[len(trainPredict) + len(testPredict) + (past_period * 3) + 2: len(normalized_data_2023_2024)-2, :] = predict24


# Plot baseline and predictions
plt.plot(minMaxScaler.inverse_transform(normalized_data_2023_2024), color='brown', label='Historical Price')
plt.plot(train_pred_plot, color='orange', label='Train Predicted 2023 Price')
plt.plot(test_pred_plot, color='green', label='Test Predicted 2023 Price')
plt.plot(pred24_plot, color='blue', label='Predicted 2024 Price')
plt.title('USD/TRY Daily Prices')
plt.xlabel('Days')
plt.ylabel('Price (TRY)')
plt.legend()
plt.show()
