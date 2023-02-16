# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:43:37 2022

@author: paulo
"""

#Neste caso pegamos a saida do Random Forest II e submetemos para o Arima

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
!pip install pmdarima
base = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/teste2009.csv',sep=';')
base.head()
base.tail()

base.Data = pd.to_datetime(base.Data)
base.set_index('Data', inplace=True)

base['Nivel'] = base['Nivel'].str.replace(',', '.')
base['Nivel'] = base['Nivel'].str.replace(',', '.').astype(float)

media = base['Nivel'].mean()
base.fillna(media,inplace=True)
base['Nivel'].isnull().sum()

train_size = int(len(base.Nivel))
train_size

train_set = base.Nivel[:10057]
test_set = base.Nivel[10057:]

len(test_set)

plt.plot(train_set)
plt.plot(test_set)
plt.plot(prediction)

#Com o AutoArima

import pmdarima as pm
auto_arima = pm.auto_arima(train_set, stepwise=False, seasonal=False)
auto_arima

forecast_test_auto = auto_arima.predict(n_periods=len(test_set))
base['forecast_manual'] = [None]*len(train_set) + list(forecast_test_auto)
base.plot()

forecast_test_auto.forecast(20)[0]

plt.figure(figsize=(20,6))
plt.plot(train_set, label = 'Training')
plt.plot(forecast_test_auto, label = 'Test')
plt.plot(prediction, label = 'Predictions')
plt.legend();



from pmdarima.arima import auto_arima
model = auto_arima(base)
model.order

model2 = auto_arima(train_set, suppress_warnings=True)

prediction = pd.DataFrame(model2.predict(n_periods=4311), index=test_set.index)
prediction.columns = ['Nivel']
prediction
pre = pd.DataFrame(prediction)
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/rf2/saidarimafinal.csv",sep=';')


test_set




from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(base,order=(2,1,2))
resultado = model.fit()
resultado.forecast(test_set)








resultado.forecast(30)[0]
resultado.forecast()

plt.figure(figsize=(20, 6))
plt.plot(base['Nivel'])
plt.plot(resultado.predict(typ='levels'))
plt.plot


plt.title('',fontsize=15)
plt.xlabel('Tempo',fontsize=15)
plt.ylabel('mm',fontsize=15)
plt.legend()
plt.show


plt.plot(base)
plt.plot(resultado.forecast(20)[0])
pre = pd.DataFrame(resultado.forecast(30)[0])
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/rf2/saidaprevisoes30.csv",sep=';')




from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from math import sqrt
mean_squared_error(test_set , forecast_test_auto)
mean_absolute_error(test_set, forecast_test_auto)
#r2_score(y_teste, previsoes)

print("Erro médio quadrático",sqrt(mean_squared_error(test_set ,forecast_test_auto)))
print("Erro médio absoluto",mean_absolute_error(test_set, forecast_test_auto))
print("Desvio Padrão",explained_variance_score(test_set ,forecast_test_auto))




