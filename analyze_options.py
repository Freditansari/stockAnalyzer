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


def analyze_options(stock_ticker, option_date_index=0):
    stock_data = yf.Ticker(stock_ticker)
    opt = stock_data.option_chain(stock_data.options[int(option_date_index)])
    calls = opt[0]
    puts = opt[1]
    totalVolumeCalls = calls['volume'].sum()
    totalVolumePuts = puts['volume'].sum()
    totalInterestCalls = calls['openInterest'].sum()
    totalInterestPuts = puts['openInterest'].sum()

    trace1 = go.Bar(
        x=calls['strike'],
        y=calls['volume'],
        # y=calls['openInterest']
        name='Calls',
        # marker_color="red"
    )

    trace2 = go.Bar(
        x=puts['strike'],
        # y=calls['volume']
        y=puts['volume'],
        name='Puts'
    )

    calls_chart = make_subplots(rows=2, cols=1)
    calls_chart.add_trace(trace1, row=1, col=1)
    calls_chart.add_trace(trace2, row=2, col=1)
    calls_chart.update_layout()

    calls_chart_output = json.dumps(calls_chart, cls=plotly.utils.PlotlyJSONEncoder)

    put_trace_volume = go.Bar(
        x=calls['strike'],
        y=calls['openInterest'],
        # y=puts['openInterest']
        name='Calls',

    )

    put_trace_openInterest = go.Bar(
        x=puts['strike'],
        # y=puts['volume']
        y=puts['openInterest'],
        name='Puts',

    )

    puts_chart = make_subplots(rows=2, cols=1)
    puts_chart.add_trace(put_trace_volume, row=1, col=1)
    puts_chart.add_trace(put_trace_openInterest, row=2, col=1)
    puts_chart.update_layout()

    puts_chart_output = json.dumps(puts_chart, cls=plotly.utils.PlotlyJSONEncoder)

    total_volumes_calls_and_puts =totalVolumeCalls + totalVolumePuts
    volumes_calls_percentage = (totalVolumeCalls/total_volumes_calls_and_puts)*100
    volumes_puts_percentage = (totalVolumePuts / total_volumes_calls_and_puts) * 100
    total_interest_puts_and_calls = totalInterestCalls + totalInterestPuts
    interest_calls_percentage = (totalInterestCalls/total_interest_puts_and_calls) * 100
    interest_puts_percentage = (totalInterestPuts/total_interest_puts_and_calls)*100


    options_metrics ={
        "totalVolumeCalls": f'{totalVolumeCalls:,} - {round(volumes_calls_percentage,2)}',
        "totalVolumePuts": f'{totalVolumePuts:,} - {round(volumes_puts_percentage,2)}',
        "totalInterestCalls" : f'{totalInterestCalls:,} - {round(interest_calls_percentage,2)}',
        "totalInterestPuts" : f'{totalInterestPuts:,} - {round(interest_puts_percentage,2)}'
    }
    options_result = (calls_chart_output, puts_chart_output, options_metrics)

    return options_result
    # puts_chart.show(renderer='colab')
