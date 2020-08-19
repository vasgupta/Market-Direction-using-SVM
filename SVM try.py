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
import time

api = fxcmpy.fxcmpy(config_file='fxcm.cfg')
symbol = 'GBP/JPY'
#data retrieval

model = svm.SVC(C=100)
isBuy = True
count = 0
#cols = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']
open_positions = api.get_open_positions()
already_Traded = False
if open_positions['currency'] is symbol:
    already_Traded = True

while True:
    candles = api.get_candles(symbol, period='D1', number=600)
    candles_min = api.get_candles(symbol, period = 'm1', number = 1)
    #candles = pd.read_csv('USD_JPY Historical Data.csv')
    candles = candles.append(candles_min)
    print(candles['bidopen'].iloc[len(candles) -1])
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

    
    model.fit(data[[RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, 'divergence']], data['direction'])
    #print(data[[RSI_column_name, SMA_column_name1, 
     #           SMA_column_name2, SMA_column_name3, 'divergence']].iloc[len(data) -1])
    X = data[[RSI_column_name, SMA_column_name1, 
                SMA_column_name2, SMA_column_name3, 'divergence']]

    #pred = model.predict(X)

    tommorrow_prediction = model.predict(X.iloc[len(X)-1: len(X)])
    print(tommorrow_prediction)
    #candles.drop([candles.index[len(candles) - 1]])
    
    
    if tommorrow_prediction == 1:
        if isBuy is False:
            api.close_all_for_symbol(symbol)
            count = 0
        if count is 0 and already_Traded is False:
            order = api.create_market_buy_order(symbol, 10)
            open_positions = open_positions.append(api.get_open_positions())
            
            count += 1
        isBuy = order.get_isBuy()
        
    if tommorrow_prediction == -1:
        if isBuy is True:
            api.close_all_for_symbol(symbol)
            count = 0
        if count is 0 and already_Traded is False:
            order = api.create_market_sell_order(symbol, 10)
            open_positions = open_positions.append(api.get_open_positions())
            count += 1
        
        isBuy = order.get_isBuy()
    open_positions.to_csv("Trades.csv", index=False)    
    time.sleep(60)

