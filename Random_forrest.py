# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:52:24 2018

@author: Vasudev the Great
"""

from sklearn.ensemble import RandomForestClassifier
import fxcm_ti
import fxcmpy
import numpy as np


api = fxcmpy.fxcmpy(config_file='fxcm.cfg')
symbol = 'GBP/JPY'

candles = api.get_candles(symbol, period='D1', number=600)

#candles_min = api.get_candles(symbol, period = 'm1', number = 1)

#candles = pd.read_csv('USD_JPY Historical Data.csv')
#candles = candles.append(candles_min, ignore_index=True)
#initialize an object of technical indicator
ta = fxcm_ti.technical_indicators(candles)

RSI_column_name = ta.add_rsi('bidopen', 14)
SMA_column_name1 = ta.add_sma('bidopen', 10)
SMA_column_name2 = ta.add_sma('bidopen', 50)
SMA_column_name3 = ta.add_sma('bidopen', 100)
EWMA_column_name1 = ta.add_ewma('bidopen', 12)
EWMA_column_name2 = ta.add_ewma('bidopen', 26)
MACD_column_name = ta.add_macd('bidopen', 12, 26)
MACD_Signal_name = ta.add_macd_signal('bidopen', 12, 26, 9)

data = candles[['bidopen', 'bidclose', RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, 
                EWMA_column_name1, EWMA_column_name2, 
                MACD_column_name, MACD_Signal_name]]

data['diff'] = data['bidclose'] - data['bidopen']
# ignore the first 100 data points, as their SMA is NAN
data = data.iloc[100:len(data)]
data['direction'] = np.sign(data['diff'])
data['sma1_direction'] = np.sign(data['bidopen']- data[SMA_column_name1])
data['sma2_direction'] = np.sign(data['bidopen'] - data[SMA_column_name2])
data['sma3_direction'] = np.sign(data['bidopen'] - data[SMA_column_name3])
# converting MACD to a category above or below signal line
data['divergence'] = data[MACD_column_name] - data[MACD_Signal_name]
data['divergence'] = np.sign(data['divergence'])

train_data = data.iloc[1:int(len(data)*0.8)]
x_train = train_data[[RSI_column_name, 'sma1_direction', 
                'sma2_direction', 'sma3_direction', 'divergence']]
y_train = train_data[['direction']]

test_data = data.iloc[int(len(data)*0.8):len(data)]
x_test = test_data[[RSI_column_name, 'sma1_direction', 
                'sma2_direction', 'sma3_direction', 'divergence']]
y_test = test_data[['direction']]

Forest = RandomForestClassifier(n_estimators = 100)
Forest = Forest.fit(x_train, y_train)

result = Forest.predict(x_test)