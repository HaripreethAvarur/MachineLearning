'''-----Imports-----'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import keras
#from datetime import datetime





'''-----Pre-Processing-----'''
df = pd.read_csv("D:\\myProjects\\Time_Series_Forecasting_LSTM_Stock_Prediction\\Dataset\\GOOG_Days_9_20-9_23.csv") #Reading CSV
#print(df.head()) 

train_dates = pd.to_datetime(df['Date']) #Seprating Dates for Plotting Later
#print(train_dates.tail(15)) 

cols = list(df)[1:6] #Only Open, High, Low, Close, Adj.Close; Date and Volume is not used
#print(cols)

df_for_training = df[cols].astype(np.float64) #creating a df with only the required cols and converting them into float so we don't lose any data when working on the values
#df_for_training.plot.line()

#LSTM uses sigmoid and tanh functions; while using these functions, the values must be normalized
df_for_training_scaled = StandardScaler().fit_transform(df_for_training)
#scaler = StandardScaler() #using standard scaler for normalization
#scaler = scaler.fit(df_for_training)
#df_for_training_scaled = scaler.transform(df_for_training)

#LSTM networks require reshaping an input data into n_samples x timesteps x n_features. 
#Ex. n_features = 5, timesteps = 14 (timesteps - number of past days used for training). 

trainX = []
trainY = []

n_future = 1 #Number of days to look into the future based on the past days.
n_past = 14 #Number of past to use to predict the future.

#Reformat input data to shape (n_samples x timesteps x n_features)
#Train-Test-Split
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))





'''-----Model-----'''
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
#model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(224, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(16, activation = "relu"))
#model.add(Dropout(0.2))
model.add(Dense(1))


from keras.optimizers import Adam

opt = Adam(learning_rate = 0.01)
model.compile(optimizer=opt, loss=keras.losses.MeanSquaredLogarithmicError()) #loss function is mean-square 
model.summary()

 
rlrop = keras.callbacks.ReduceLROnPlateau()

#Fitting the model
history = model.fit(trainX, trainY, epochs=50, validation_split=0.2, callbacks = [rlrop])





'''-----Visualization-----'''
#Plotting
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.xlim(0,3)
plt.show()

#Forecasting 
n_future_prediction = 30
n_past_days = 1

predict_period_dates = pd.date_range(list(train_dates)[-n_past_days], periods=n_future_prediction, freq='1d').tolist()
print(predict_period_dates)

#Predict
prediction = model.predict(trainX[-n_future_prediction:]) #shape = (n, 1)

#Inverse Transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = StandardScaler().fit(df_for_training).inverse_transform(prediction_copies)[:,0]

'''
# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2022-9-10']

sns.lineplot(original['Date'], original['Open'])
sns.lineplot(df_forecast['Date'], df_forecast['Open'])
'''

forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

original = df[['Date', 'Open']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2023-05-10']

# Plot original data first, and then the forecast data
sns.lineplot(data=original, x='Date', y='Open', label='Original')
sns.lineplot(data=df_forecast, x='Date', y='Open', label='Forecast')

plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Original vs. Forecast Open Prices')
plt.legend()
plt.show()
