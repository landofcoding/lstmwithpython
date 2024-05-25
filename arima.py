# LSTM for USD/TRY daily price prediction with window regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load USD/TRY data for 2023 and 2024
data_2023_2024 = read_csv('usdtry23-24.csv', usecols=[1], engine='python')

# Convert data to float and normalize
price_data_2023_2024 = data_2023_2024.values.astype('float32')

price_data_2023 = price_data_2023_2024[:364]
price_data_2024 = price_data_2023_2024[365:]

# Split into training dataset and testing
trainDataSize = int(len(price_data_2023) * 0.90)
testDataSize = len(price_data_2023) - trainDataSize
trainDataset, testDataset = price_data_2023[0:trainDataSize, :], price_data_2023[trainDataSize:len(price_data_2023), :]
pastDataset = [x for x in trainDataset]
predictedDataset2023 = list()

# walk-forward validation
for t in range(len(testDataset)):
 arimamodel = ARIMA(pastDataset, order=(5,3,0))
 model_fit = arimamodel.fit()
 prediction = model_fit.forecast()
 prd = prediction[0]
 predictedDataset2023.append(prd)
 exp = testDataset[t]
 pastDataset.append(exp)
 print('prediction=%f, expected=%f' % (prd, exp))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(testDataset, predictedDataset2023))
print('Test RMSE: %.2f' % rmse)

predictedDataset2024 = list()

for z in range(len(price_data_2024)):
 arimamodel = ARIMA(pastDataset, order=(5,3,0))
 model_fit = arimamodel.fit()
 prediction = model_fit.forecast()
 prd = prediction[0]
 predictedDataset2024.append(prd)
 exp = testDataset[t]
 pastDataset.append(exp)
 print('prediction=%f, expected=%f' % (prd, exp))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(price_data_2024, predictedDataset2024))
print('2024 RMSE: %.2f' % rmse)

plt.plot(testDataset)
plt.plot(predictedDataset2023, color='red')

plt.plot(testDataset, color='brown', label='Actual Test Prices')
plt.plot(predictedDataset2023, color='green', label='Test Predicted 2023 Price')
plt.title('USD/TRY Daily Price')
plt.xlabel('Days')
plt.ylabel('Price (TRY)')
plt.legend()
plt.show()
