# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:10:52 2022

@author: paulo
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
base = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/modelo0105.csv',sep=';')
base.head()
base.tail()

base.Data = pd.to_datetime(base.Data)
base.set_index('Data', inplace=True)

base['Nivel'] = base['Nivel'].str.replace(',', '.')
base['Nivel'] = base['Nivel'].str.replace(',', '.').astype(float)

base.dropna()


base.isnull().sum()
          
          
media = base['Nivel'].mean()
base.fillna(media,inplace=True)
base['Nivel'].isnull().sum()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(base, period=12)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.plot(seasonal)


res = pd.DataFrame(seasonal)
res.to_csv("C:/Users/paulo/Documents/base4/sonivel/seasonal.csv",sep=';')

res = pd.DataFrame(trend)
res.to_csv("C:/Users/paulo/Documents/base4/sonivel/tendencia.csv",sep=';')

res = pd.DataFrame(residual)
res.to_csv("C:/Users/paulo/Documents/base4/sonivel/residuo.csv",sep=';')

fig = plt.figure(figsize=(8,6))
fig = decomposition.plot()

base.Nivel.rolling(12).mean().plot(figsize=(15,6))

base.Nivel.groupby(base.index.year).sum().plot(figsize=(15,6))

filtro = (base.index.year >= 1990) & (base.index.year <= 1995)
base[filtro].Nivel.diff().plot(figsize=(15,6))

base.Nivel.diff(10).groupby(base.index.year).sum().plot(figsize=(15,6))


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(base,order=(2,1,2))
resultado = model.fit()
resultado

resw = pd.DataFrame(resultado.predict(typ='levels'))
resw.to_csv("C:/Users/paulo/Documents/base4/sonivel/predictarima.csv",sep=';')
resultado.predict(start='2018-06-01', end='2018-06-01')


(resultado.resid **2).mean()

resultado.resid.plot(kind='box')



res = pd.DataFrame(resultado.resid)
res.to_csv("C:/Users/paulo/Documents/base4/sonivel/residuonovo.csv",sep=';')

plt.figure(figsize=(20, 6))
plt.plot(base['Nivel'])
plt.plot(resultado.predict(typ='levels'))
plt.plot


plt.title('',fontsize=15)
plt.xlabel('Tempo',fontsize=15)
plt.ylabel('mm',fontsize=15)
plt.legend()
plt.show


resultado.forecast(1)[0]

resultado.forecast(30)[0].mean()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
plot_acf(base['Nivel'])
plot_pacf(base['Nivel'])


#-----------------------------------------------------------------------
len(base)

train_set = base.Nivel[:10057]
test_set = base.Nivel[10057:]

import pmdarima as pm
auto_arima = pm.auto_arima(train_set, stepwise=False, seasonal=False)
auto_arima

forecast_test_auto = auto_arima.predict(n_periods=len(test_set))
base['forecast_manual'] = [None]*len(train_set) + list(forecast_test_auto)

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from math import sqrt
mean_squared_error(test_set , forecast_test_auto)
mean_absolute_error(test_set, forecast_test_auto)
#r2_score(y_teste, previsoes)
explained_variance_score(test_set ,forecast_test_auto)

print("Erro médio quadrático",sqrt(mean_squared_error(test_set ,forecast_test_auto)))
print("Erro médio absoluto",mean_absolute_error(test_set, forecast_test_auto))
print("Desvio Padrão",explained_variance_score(test_set ,forecast_test_auto))




