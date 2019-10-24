import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation



df = pd.read_csv('avocado.csv')
df = df.sort_values(by='Date')


#Abacoda type -> conventional, Abaconda region = TotalUs
df = df.loc[(df.type == 'conventional') & (df.region == 'TotalUS')]

#string type -> datetime
df.sort_values(by=['Date'])

df['Date'] = pd.to_datetime(df['Date'])


#initialize reset_index
data = df[['Date','AveragePrice']].reset_index(drop=True)


print(data.head())
data = data.rename(columns = {'Date' : 'ds', 'AveragePrice' : 'y'})


AP = data['y'].values.tolist()
print("AP",AP)
print("AP_len", len(AP))

x_data = []
y_data = []


seq_len = 10

for index in range(0, len(AP)-seq_len):
     x_data.append(AP[index:index + seq_len])
     y_data.append(AP[index + seq_len])

x_data = np.array(x_data)
y_data = np.array(y_data)

row = int(round(y_data.shape[0])*0.8)
total_row = int(y_data.shape[0])

print("row",row)
print("total_row",total_row)

print(x_data.shape)
print(y_data.shape)


x_train = x_data[:row,:]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


print(x_train.shape)


x_test =  x_data[row:total_row]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


y_train = y_data[:row]
y_test =  y_data[row:total_row]


model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(10, 1)))
model.add(LSTM(30, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=10)


pred = model.predict(x_test)
pred = np.squeeze(pred)


print("prediction value \n",  pred)
print("ground truth value \n", y_test)
