import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('avocado.csv')
df = df.sort_values(by='Date')
print(df.head())


print(df.describe())


print(df.groupby('type').mean())

#Abacoda type -> conventional, Abaconda region = TotalUs
df = df.loc[(df.type == 'conventional') & (df.region == 'TotalUS')]

#string type -> datetime
df.sort_values(by=['Date'])

df['Date'] = pd.to_datetime(df['Date'])


#initialize reset_index
data = df[['Date','AveragePrice']].reset_index(drop=True)


print(data.head())
data = data.rename(columns = {'Date' : 'ds', 'AveragePrice' : 'y'})


plt.plot(data.ds, data.y,(16,8))

model = Prophet()
model.fit(data)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print(forecast.tail())

fig1 = model.plot(forecast)
#plt.show()


fig2 = model.plot_components(forecast)
#plt.show()