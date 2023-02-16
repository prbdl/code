# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:02:16 2022

@author: paulo
"""

#NESTE EXPERIMENTO, PEGAMOS A PREDICAO DO RF1 COM A PREDICAO DO ARIMA
#JUNTO COM A BASE REAL

import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns

base = pd.read_csv('C:/Users/paulo/Documents/base4/sonivel/rf2/rf3.csv',sep=';')
base

base['Real'] = base['Real'].str.replace(',', '.')
base['Real'] = base['Real'].str.replace(',', '.').astype(float)

base['Na'] = base['Na'].str.replace(',', '.')
base['Na'] = base['Na'].str.replace(',', '.').astype(float)

base['Nr'] = base['Nr'].str.replace(',', '.')
base['Nr'] = base['Nr'].str.replace(',', '.').astype(float)


base.dropna()
media = base['Real'].mean()
base.fillna(media,inplace=True)
base['Real'].isnull().sum()

base.dropna()
media = base['Na'].mean()
base.fillna(media,inplace=True)
base['Na'].isnull().sum()

base.dropna()
media = base['Nr'].mean()
base.fillna(media,inplace=True)
base['Nr'].isnull().sum()

previsores = base.iloc[:, 0:2].values
real = base.iloc[:, 2].values
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
pre.to_csv("C:/Users/paulo/Documents/base4/sonivel/rf2/previsoesrf2.csv",sep=';')

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




