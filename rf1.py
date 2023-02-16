# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:17:45 2022

@author: paulo
"""

#NESTE EXPERIMENTO, PEGAMOS O RESIDUO DO ARIMA (NIVEL DE AGUA)
#E ADCIONAMOS AS VARIAVEIS SATELITE JUNTO COM A BASE REAL

import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns

base = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/rf1/modelo0105.csv',sep=';')
base.head

base['Evaporacao do Pichemm'] = base['Evaporacao do Pichemm'].str.replace(',', '.')
base['Evaporacao do Pichemm'] = base['Evaporacao do Pichemm'].astype(float)

base['Insolocao totalh'] = base['Insolocao totalh'].str.replace(',', '.')
base['Insolocao totalh'] = base['Insolocao totalh'].astype(float)

base['Precipitação Taotalmm'] = base['Precipitação Taotalmm'].str.replace(',', '.')
base['Precipitação Taotalmm'] = base['Precipitação Taotalmm'].astype(float)

base['Temperatura MaximaC'] = base['Temperatura MaximaC'].str.replace(',', '.')
base['Temperatura MaximaC'] = base['Temperatura MaximaC'].astype(float)

base['Temperatura MediaC'] = base['Temperatura MediaC'].str.replace(',', '.') #
base['Temperatura MediaC'] = base['Temperatura MediaC'].str.replace(',', '.').astype(float)

base['Temperatura minimaC'] = base['Temperatura minimaC'].str.replace(',', '.')
base['Temperatura minimaC'] = base['Temperatura minimaC'].str.replace(',', '.').astype(float)

base['Umidade do relativa do ar%'] = base['Umidade do relativa do ar%'].str.replace(',', '.')
base['Umidade do relativa do ar%'] = base['Umidade do relativa do ar%'].str.replace(',', '.').astype(float)

base['Velocidade media vento%'] = base['Velocidade media vento%'].str.replace(',', '.')
base['Velocidade media vento%'] = base['Velocidade media vento%'].str.replace(',', '.').astype(float)

base['Nivel'] = base['Nivel'].str.replace(',', '.')
base['Nivel'] = base['Nivel'].str.replace(',', '.').astype(float)

base['Nr'] = base['Nr'].str.replace(',', '.')
base['Nr'] = base['Nr'].str.replace(',', '.').astype(float)


base.isnull().sum()


media = base['Nivel'].mean()
base.fillna(media,inplace=True)
base['Nivel'].isnull().sum()

media = base['Velocidade media vento%'].mean()
base.fillna(media,inplace=True)
base['Velocidade media vento%'].isnull().sum()

media = base['Umidade do relativa do ar%'].mean()
base.fillna(media,inplace=True)
base['Umidade do relativa do ar%'].isnull().sum()

media = base['Temperatura minimaC'].mean()
base.fillna(media,inplace=True)
base['Temperatura minimaC'].isnull().sum()

media = base['Temperatura MediaC'].mean()
base.fillna(media,inplace=True)
base['Temperatura MediaC'].isnull().sum()

media = base['Temperatura MaximaC'].mean()
base.fillna(media,inplace=True)
base['Temperatura MaximaC'].isnull().sum()

media = base['Precipitação Taotalmm'].mean()
base.fillna(media,inplace=True)
base['Precipitação Taotalmm'].isnull().sum()

media = base['Insolocao totalh'].mean()
base.fillna(media,inplace=True)
base['Insolocao totalh'].isnull().sum()

media = base['Evaporacao do Pichemm'].mean()
base.fillna(media,inplace=True)
base['Evaporacao do Pichemm'].isnull().sum()

previsores = base.iloc[:, 0:9].values
real = base.iloc[:, 9].values
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
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/rf1/previsoesrf1.csv",sep=';')


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 6))
plt.plot(y, color = 'red', label = 'Real')
plt.plot(previsoes, color = 'blue', label= 'Previsões')

plt.title('',fontsize=15)
plt.xlabel('Tempo',fontsize=15)
plt.ylabel('mm',fontsize=15)
plt.legend(fontsize=15)
plt.show








