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
import plotly

def analyze_stock(stock_ticker):
    end = datetime.datetime.now()#-timedelta(7) #remove timedetla -1 to go to productions mode
    start = datetime.datetime.now()-timedelta(365*20)

    # stock_ticker="AMZN"
    stock=yf.download(stock_ticker, start, end)
    '''
    resample daily to weekly data. 
    add changes open-high open-low column
    '''
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

    trace1 =go.Scatter(
    x=open_low_forecast['ds'],
    y=open_low_forecast['yhat'] ,
    name='[low band]average low forecast',

    )
    trace2= go.Scatter(
    x=open_low_forecast['ds'],
    y=open_low_forecast['yhat_lower'],
    name='[low band]lowest forecast'
    )
    trace3= go.Scatter(
    x=open_low_forecast['ds'],
    y=open_low_forecast['yhat_upper'],
    name='[low band]highest low forecast',
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    line_color='rgba(255,255,255,0)',
    )
    trace4= go.Scatter(
    x=training_open_low['ds'],
    y=training_open_low['y'],
    name='[low band]actual low',
    mode='markers'
    )

    # high bands

    trace5 =go.Scatter(
    x=open_high_forecast['ds'],
    y=open_high_forecast['yhat'] ,
    name='[high band]average high forecast',

    )
    trace6= go.Scatter(
    x=open_high_forecast['ds'],
    y=open_high_forecast['yhat_lower'],
    name='[high band]highest forecast'
    )
    trace7= go.Scatter(
    x=open_high_forecast['ds'],
    y=open_high_forecast['yhat_upper'],
    name='[high band]highest low forecast',
    fill='tonexty',
    # fillcolor='rgba(0,100,80,0.2)',
    # line_color='rgba(255,255,255,0)',
    )
    trace8= go.Scatter(
    x=training_open_high['ds'],
    y=training_open_high['y'],
    name='[high band]actual high',
    mode='markers'
    )
    # end of high bands

    high_low_chart= make_subplots(specs=[[{"secondary_y": True}]])
    #  Low Bands
    high_low_chart.add_trace(trace1)
    high_low_chart.add_trace(trace2)
    high_low_chart.add_trace(trace3)
    high_low_chart.add_trace(trace4)

    high_low_chart.add_trace(trace5)
    high_low_chart.add_trace(trace6)
    high_low_chart.add_trace(trace7)
    high_low_chart.add_trace(trace8)

    high_low_chart.update_layout(height=600)

    high_low_chart_output = json.dumps(high_low_chart, cls=plotly.utils.PlotlyJSONEncoder)

    current_price = weekly_stock['Open'].tail(1).values
    date_today = datetime.datetime.now()
    copy_open_low_forecast = open_low_forecast.set_index('ds')
    this_week_forecast_index = copy_open_low_forecast.index.get_loc(date_today, method='nearest')
    low_forecast_percentage = open_low_forecast.iloc[this_week_forecast_index]

    copy_open_high_forecast = open_high_forecast.set_index('ds')
    this_week_high_forecast_index = copy_open_high_forecast.index.get_loc(date_today, method='nearest')
    high_forecast_percentage = open_high_forecast.iloc[this_week_high_forecast_index]

    lowest_forecast = current_price[0] + (current_price[0] * (low_forecast_percentage['yhat_lower'] / 100))
    average_low_forecast = current_price[0] + (current_price[0] * (low_forecast_percentage['yhat'] / 100))

    highest_forecast = current_price[0] + (current_price[0] * (high_forecast_percentage['yhat_upper'] / 100))
    average_high_forecast = current_price[0] + (current_price[0] * (high_forecast_percentage['yhat'] / 100))

    # estimates = f'''
    #     estimate :
    #     ticker : {stock_ticker}
    #     ****Weekly estimate ****
    #     open price : {current_price[0]},
    #     low estimate : {round(lowest_forecast, 2)} - {round(average_low_forecast, 2)}
    #     high estimate : {round(average_high_forecast, 2)} - {round(highest_forecast, 2)}
    #     ********************** \n
    #     '''
    estimates ={
        "open_price": str(round(current_price[0],2)) ,
        "low_estimate": str( round(lowest_forecast, 2) )+ " - " + str(round(average_low_forecast, 2)),
        "high_estimate" : str( round(average_high_forecast, 2) )+ " - " + str(round(highest_forecast, 2))
    }
    # estimates= json.dumps(estimates)

    changes_training_df = pd.DataFrame()
    changes_training_df['ds'] = pd.to_datetime(train_set.index).tz_localize(None)
    changes_training_df['y'] = train_set['changes'].values
    changes_model = Prophet(weekly_seasonality=False)
    changes_model.fit(changes_training_df)

    changes_future = changes_model.make_future_dataframe(52, freq='W')
    changes_forecast = changes_model.predict(changes_future)

    trace2 = go.Line(
        x=changes_forecast['ds'].tail(100),
        y=changes_forecast['yearly'].tail(100),
        name='yearly',

    )

    trend_figure = make_subplots(rows=1, cols=1)
    trend_figure.add_trace(trace2, row=1, col=1)
    trend_chart_output = json.dumps(trend_figure, cls=plotly.utils.PlotlyJSONEncoder)

    final_result = (trend_chart_output, estimates,high_low_chart_output)
    return final_result

