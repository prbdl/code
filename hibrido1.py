# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 06:54:41 2023

@author: paulo
"""

##ARIMA I

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
base = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/hibrido1/nivel.csv',sep=';')

base.Data = pd.to_datetime(base.Data)
base.set_index('Data', inplace=True)

base['Nivel'] = base['Nivel'].str.replace(',', '.')
base['Nivel'] = base['Nivel'].str.replace(',', '.').astype(float)

base.dropna()


base.isnull().sum()
          
          
media = base['Nivel'].mean()
base.fillna(media,inplace=True)
base['Nivel'].isnull().sum()

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(base,order=(2,1,2))
resultado = model.fit()
resultado

resw = pd.DataFrame(resultado.predict(typ='levels'))
##Predicao Arima I
resw.to_csv("C:/Users/paulo/Documents/base4/sonivel/hibrido1/predictarima.csv",sep=';')

##Residuos
res = pd.DataFrame(resultado.resid)
res.to_csv("C:/Users/paulo/Documents/base4/sonivel/hibrido1/residuonovo.csv",sep=';')

##Random Forest I

#NESTE EXPERIMENTO, PEGAMOS O RESIDUO DO ARIMA (NIVEL DE AGUA)
#E ADCIONAMOS AS VARIAVEIS SATELITE JUNTO COM A BASE REAL

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns

base1 = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/hibrido1/satelite.csv',sep=';')
base1.head

base1['Evaporacao do Pichemm'] = base1['Evaporacao do Pichemm'].str.replace(',', '.')
base1['Evaporacao do Pichemm'] = base1['Evaporacao do Pichemm'].astype(float)

base1['Insolocao totalh'] = base1['Insolocao totalh'].str.replace(',', '.')
base['Insolocao totalh'] = base['Insolocao totalh'].astype(float)

base1['Precipitação Taotalmm'] = base1['Precipitação Taotalmm'].str.replace(',', '.')
base1['Precipitação Taotalmm'] = base1['Precipitação Taotalmm'].astype(float)

base1['Temperatura MaximaC'] = base1['Temperatura MaximaC'].str.replace(',', '.')
base1['Temperatura MaximaC'] = base1['Temperatura MaximaC'].astype(float)

base1['Temperatura MediaC'] = base1['Temperatura MediaC'].str.replace(',', '.') #
base1['Temperatura MediaC'] = base1['Temperatura MediaC'].str.replace(',', '.').astype(float)

base1['Temperatura minimaC'] = base1['Temperatura minimaC'].str.replace(',', '.')
base1['Temperatura minimaC'] = base1['Temperatura minimaC'].str.replace(',', '.').astype(float)

base1['Umidade do relativa do ar%'] = base1['Umidade do relativa do ar%'].str.replace(',', '.')
base1['Umidade do relativa do ar%'] = base1['Umidade do relativa do ar%'].str.replace(',', '.').astype(float)

base1['Velocidade media vento%'] = base1['Velocidade media vento%'].str.replace(',', '.')
base1['Velocidade media vento%'] = base1['Velocidade media vento%'].str.replace(',', '.').astype(float)

##Nivel e o residuo de Nivel

base1['Nivel'] = base1['Nivel'].str.replace(',', '.')
base1['Nivel'] = base1['Nivel'].str.replace(',', '.').astype(float)

#Nr e o Nivel Real
base1['Nr'] = base1['Nr'].str.replace(',', '.')
base1['Nr'] = base1['Nr'].str.replace(',', '.').astype(float)


base1.isnull().sum()


media = base1['Nivel'].mean()
base1.fillna(media,inplace=True)
base1['Nivel'].isnull().sum()

media = base1['Velocidade media vento%'].mean()
base1.fillna(media,inplace=True)
base1['Velocidade media vento%'].isnull().sum()

media = base1['Umidade do relativa do ar%'].mean()
base1.fillna(media,inplace=True)
base1['Umidade do relativa do ar%'].isnull().sum()

media = base1['Temperatura minimaC'].mean()
base1.fillna(media,inplace=True)
base1['Temperatura minimaC'].isnull().sum()

media = base1['Temperatura MediaC'].mean()
base1.fillna(media,inplace=True)
base1['Temperatura MediaC'].isnull().sum()

media = base1['Temperatura MaximaC'].mean()
base1.fillna(media,inplace=True)
base1['Temperatura MaximaC'].isnull().sum()

media = base1['Precipitação Taotalmm'].mean()
base1.fillna(media,inplace=True)
base1['Precipitação Taotalmm'].isnull().sum()

media = base1['Insolocao totalh'].mean()
base1.fillna(media,inplace=True)
base1['Insolocao totalh'].isnull().sum()

media = base1['Evaporacao do Pichemm'].mean()
base1.fillna(media,inplace=True)
base1['Evaporacao do Pichemm'].isnull().sum()

previsores = base1.iloc[:, 0:9].values
real = base1.iloc[:, 9].values
previsores
real

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
x = normalizador.fit_transform(previsores)
x
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
y = normalizador_previsao.fit_transform(real.reshape(-1,1))

from sklearn.ensemble import RandomForestRegressor

regressor_random_forest = RandomForestRegressor(n_estimators=200,max_depth=6,random_state=123,min_samples_leaf=3)
regressor_random_forest.fit(x,y)

previsoes = regressor_random_forest.predict(previsores)
previsoes = normalizador_previsao.inverse_transform(previsoes.reshape(1,-1))
previsoes
previsoes.mean()

y = normalizador_previsao.inverse_transform(y.reshape(1,-1))
y.mean()


pre = pd.DataFrame(previsoes)
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/hibrido1/previsoesrf1.csv",sep=';')

##Aqui inicia o RF2

#NESTE EXPERIMENTO, PEGAMOS A PREDICAO DO RF1 COM A PREDICAO DO ARIMA
#JUNTO COM A BASE REAL

base2 = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/hibrido1/rf2.csv',sep=';')
base

base2['Real'] = base2['Real'].str.replace(',', '.')
base2['Real'] = base2['Real'].str.replace(',', '.').astype(float)

#Na e a previsao Arima
base2['Na'] = base2['Na'].str.replace(',', '.')
base2['Na'] = base2['Na'].str.replace(',', '.').astype(float)

#Nr previsao RFI
base2['Nr'] = base2['Nr'].str.replace(',', '.')
base2['Nr'] = base2['Nr'].str.replace(',', '.').astype(float)


base2.dropna()
media = base2['Real'].mean()
base2.fillna(media,inplace=True)
base2['Real'].isnull().sum()

base2.dropna()
media = base2['Na'].mean()
base2.fillna(media,inplace=True)
base2['Na'].isnull().sum()

base2.dropna()
media = base2['Nr'].mean()
base2.fillna(media,inplace=True)
base2['Nr'].isnull().sum()

previsores = base2.iloc[:, 0:2].values
real = base2.iloc[:, 2].values
previsores
real

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
x = normalizador.fit_transform(previsores)
x
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
y = normalizador_previsao.fit_transform(real.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.30, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

regressor_random_forest = RandomForestRegressor(n_estimators=200,max_depth=6,random_state=123,min_samples_leaf=3)
regressor_random_forest.fit(X_treinamento,y_treinamento)

previsoes = regressor_random_forest.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes.reshape(-1,1))
previsoes
previsoes.mean()

y_teste = y_teste.reshape(-1,1)
y_teste = normalizador_previsao.inverse_transform(y_teste)

y_teste
y_teste.mean()

pre = pd.DataFrame(previsoes)
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/hibrido1/previsoesrf2.csv",sep=';')

#Resultados Finais

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from math import sqrt
mean_squared_error(y_teste, previsoes)
mean_absolute_error(y_teste, previsoes)
#r2_score(y_teste, previsoes)

print("Erro médio quadrático",sqrt(mean_squared_error(y_teste, previsoes)))
print("Erro médio absoluto",mean_absolute_error(y_teste, previsoes))
print("Desvio Padrão",explained_variance_score(y_teste, previsoes))

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.plot(y_teste, color = 'red', label = 'Real')
plt.plot(previsoes, color = 'blue', label= 'Previsões')
plt.title('',fontsize=15)
plt.xlabel('Tempo',fontsize=15)
plt.ylabel('mm',fontsize=15)
plt.legend(fontsize=15)
plt.show

##Inicio Arima Final

#Neste caso pegamos a saida do Random Forest II e submetemos para o Arima

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
!pip install pmdarima
base3 = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/hibrido1/teste2009.csv',sep=';')
base3.head()
base3.tail()

base3.Data = pd.to_datetime(base.Data)
base3.set_index('Data', inplace=True)

base3['Nivel'] = base3['Nivel'].str.replace(',', '.')
base3['Nivel'] = base3['Nivel'].str.replace(',', '.').astype(float)

media = base3['Nivel'].mean()
base3.fillna(media,inplace=True)
base3['Nivel'].isnull().sum()

train_size = int(len(base3.Nivel))
train_size

train_set = base3.Nivel[:10057]
test_set = base3.Nivel[10057:]

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
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/hibrido1/saidarimafinal.csv",sep=';')

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


