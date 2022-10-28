def analyze_stock(stock_ticker):
    import json
    import numpy as np

    import pandas_datareader.data as web
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import datetime
    from datetime import timedelta
    import threading
    import time
    import requests
    import random
    # plotly stuff
    from plotly import __version__
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.express as px
    import plotly.graph_objects as go
    import cufflinks as cf
    from plotly.subplots import make_subplots

    cf.go_offline()
    import matplotlib.pyplot as plt
    from matplotlib import style
    import yfinance as yf
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly


    end = datetime.datetime.now()#-timedelta(7) #remove timedetla -1 to go to productions mode
    start = datetime.datetime.now()-timedelta(365*20)

    stock=yf.download(stock_ticker, start, end)
    stock['Changes'] = stock['Adj Close'].pct_change()
    current_price = stock['Close'].tail(1).values

    logic = {'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Volume': 'sum'}

    logic = {'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Volume': 'sum'}

    offset = pd.DateOffset(days=-6)
    weekly_stock = stock.resample('W', loffset=offset).apply(logic)
    weekly_stock['changes'] = (weekly_stock['Close'].pct_change())*100
    # weekly_stock['open-high'] = ((weekly_stock['High']-weekly_stock['Open'])/weekly_stock['High'])
    weekly_stock['open-high'] = ((weekly_stock['High']-weekly_stock['Open'])/weekly_stock['Open'])*100
    weekly_stock['open-low'] = ((weekly_stock['Low']-weekly_stock['Open'])/weekly_stock['Open'])*100
    last_known_price = weekly_stock.tail(1)
    
    weekly_stock.dropna(inplace=True)

    train_set =weekly_stock
    
    # prepare open-low data for prophet 
    training_open_low = pd.DataFrame()
    training_open_low['ds'] =pd.to_datetime(train_set.index).tz_localize(None)
    training_open_low['y'] = train_set['open-low'].values
    
    #using prophet to predict weekly open-low      
    model = Prophet(weekly_seasonality=False)
    model.fit(training_open_low)
    
    future = model.make_future_dataframe(52, freq='W')
    open_low_forecast = model.predict(future)
    
    # prepare open-low data for prophet 
    training_open_high = pd.DataFrame()
    training_open_high['ds'] =pd.to_datetime(train_set.index).tz_localize(None)
    training_open_high['y'] = train_set['open-high'].values
    
    #using prophet to predict weekly open-high      
    model = Prophet(weekly_seasonality=False)
    model.fit(training_open_high)
    
    future = model.make_future_dataframe(52, freq='W')
    open_high_forecast = model.predict(future)
 
 
    
    # estimates =f'''

    #     estimate : \n
    #     ticker : {stock_ticker} \n
    #     ****Weekly estimate **** \n
    #     open price : {open_price}, \n
    #     low estimate : {low_estimate} \n
    #     high estimate : {high_estimate} \n
    #     ********************** \n


    #     *****Estimate from current price ***** \n
    #     current price : {current_price}, \n
    #     low estimate : {current_low_estimate} \n
    #     high estimate : {current_high_estimate} \n
    #     **************************************
    #     '''
   
    return 'final_result'

print(analyze_stock("aapl"))