# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:07:43 2018

@author: Vas
"""
from sklearn import svm
import pandas as pd
import fxcm_ti
import fxcmpy
import numpy as np

api = fxcmpy.fxcmpy(config_file='fxcm_live_token.cfg', server = 'real')
symbol = 'EUR/USD'
#data retrieval
candles = api.get_candles(symbol, period='D1', number=600)
candles_min = api.get_candles(symbol, period = 'm1', number = 1)

#candles = pd.read_csv('USD_JPY Historical Data.csv')
candles = candles.append(candles_min, ignore_index=True)
#initialize an object of technical indicator
ta = fxcm_ti.technical_indicators(candles)

RSI_column_name = ta.add_rsi('bidopen', 14)
SMA_column_name1 = ta.add_sma('bidopen', 10)
SMA_column_name2 = ta.add_sma('bidopen', 50)
SMA_column_name3 = ta.add_sma('bidopen', 100)
MACD_column_name = ta.add_macd('bidopen', 12, 26)
MACD_Signal_name = ta.add_macd_signal('bidopen', 12, 26, 9)

data = candles[['bidopen', RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, MACD_column_name, MACD_Signal_name]]

data['diff'] = data['bidopen'].diff()
data = data.iloc[100:len(candles)]
data['direction'] = np.sign(data['diff'])
# converting MACD to a category above or below signal line
data['divergence'] = data[MACD_column_name] - data[MACD_Signal_name]
data['divergence'] = np.sign(data['divergence'])

model = svm.SVC(C=100)
model.fit(data[[RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, 'divergence']], data['direction'])

X = data[[RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, 'divergence']]

pred = model.predict(X)

compare = pd.DataFrame(pred)
compare['Actual'] = data['direction']

array = data['direction'] == pred

count = 0

for row in array:
    if row == np.True_:
        count += 1

#X = X.append(X.iloc[len(X)-1], ignore_index=True)

tommorrow_prediction = model.predict(X.iloc[len(X)-1: len(X)])

'''
if tommorrow_prediction is 1:
    order = api.create_market_buy_order(symbol, 10)
if tommorrow_prediction is -1:
    order = api.create_market_sell_order(symbol, 10)
'''
