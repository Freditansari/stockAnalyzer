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
    import matplotlib.pyplot as plt
    from matplotlib import style
    import yfinance as yf
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import plotly

    from flask import jsonify
    from flask import Markup


    datetime.datetime.now()#-timedelta(7) #remove timedetla -1 to go to productions mode
    start = datetime.datetime.now()-timedelta(365*10)
    end = datetime.datetime.now()#-timedelta(7) 

    stock=yf.download(stock_ticker, start, end)
    current_price = stock['Close'].tail(1).values

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

    X = weekly_stock[["Open"]]
    y = weekly_stock[['open-low']]

    number_of_test_size = 0.4

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
    weekly_stock = clean_dataset(weekly_stock)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number_of_test_size)


    X_high = weekly_stock[["Open"]]
    y_high = weekly_stock[['open-high']]
    X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y_high, test_size=number_of_test_size)
    lm = LinearRegression()
    lm.fit(X_train,y_train)

    lm_high = LinearRegression()
    lm_high.fit(X_high_train,y_high_train)

    open_price = last_known_price['Open'].values

    predictions = lm.predict([[open_price[0]]])

    predictions_high = lm_high.predict([[open_price[0]]])
    open_low=predictions[0][0]
    open_high = predictions_high[0][0]

    low_estimate = open_price +((open_price* (open_low/100)))
    high_estimate = open_price +((open_price* (open_high/100)))

    current_low_estimate = current_price +((current_price* (open_low/100)))

    current_high_estimate = current_price +((current_price* (open_high/100)))


    estimates =f'''

        estimate : \n
        ticker : {stock_ticker} \n
        ****Weekly estimate **** \n
        open price : {open_price}, \n
        low estimate : {low_estimate} \n
        high estimate : {high_estimate} \n
        ********************** \n


        *****Estimate from current price ***** \n
        current price : {current_price}, \n
        low estimate : {current_low_estimate} \n
        high estimate : {current_high_estimate} \n
        **************************************
        '''
    lm_line = lm.predict(X)
    lm_line_high = lm_high.predict(X)

    for prediction in lm_line: 
        weekly_stock['linear_regression_low'] =prediction[0]

    for prediction in lm_line_high: 
        weekly_stock['linear_regression_high'] =prediction[0]

    
    trace1 =go.Scatter(
        x=weekly_stock.index,
        y=weekly_stock['changes'],
        name='changes',
        mode = 'markers',
    )

    trace4 =go.Line(
        x=weekly_stock.index,
        y=weekly_stock['linear_regression_low'],
        name='linear_regression_low',
    )

    trace5 =go.Scatter(
        x=weekly_stock.index,
        y=weekly_stock['open-low'],
        name='open-low',
        mode = 'markers'
    )

    trace6 =go.Scatter(
        x=weekly_stock.index,
        y=weekly_stock['open-high'],
        name='open-high',
        mode = 'markers'
    )

    trace7 =go.Line(
        x=weekly_stock.index,
        y=weekly_stock['linear_regression_high'],
        name='linear_regression_high',
    )

    figure= make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(trace1)
    # figure.add_trace(trace2)
    # figure.add_trace(trace3)
    figure.add_trace(trace4)
    figure.add_trace(trace5)
    figure.add_trace(trace6)
    figure.add_trace(trace7)


    # chart_result = figure.to_html(full_html = False, include_plotlyjs=False)
    chart_result = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # weekly_stock.query("changes<linear_regression_low").tail(14)
    few_weeks = weekly_stock
    number_of_rights = len(few_weeks[(few_weeks['changes'] >= few_weeks['linear_regression_low'])& (few_weeks['changes'] <= few_weeks['linear_regression_high'])])

    rights_percentage = (number_of_rights / len(few_weeks))*100

    confidence = round(rights_percentage,2)

    # calculate macd
    k = weekly_stock['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = weekly_stock['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k-d 
    macd_s = macd.ewm(span=9, adjust = False, min_periods = 9).mean()

    macd_h = macd- macd_s

    weekly_stock['macd'] = weekly_stock.index.map(macd)
    weekly_stock['macd_h'] = weekly_stock.index.map(macd_h)

    weekly_stock['macd_s'] = weekly_stock.index.map(macd_s)


    k = stock['Adj Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = stock['Adj Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k-d 
    macd_s = macd.ewm(span=9, adjust = False, min_periods = 9).mean()

    macd_h = macd- macd_s

    stock['macd'] = stock.index.map(macd)
    stock['macd_h'] = stock.index.map(macd_h)

    stock['macd_s'] = stock.index.map(macd_s)

    stock['gains']=stock['Adj Close'].pct_change()
    stock['cum_gain']= stock['gains'].cumsum()
    
    stock['gains']=stock['Adj Close'].pct_change()
    stock['cum_gain']= stock['gains'].cumsum()

    trace1 =go.Candlestick(
    x=weekly_stock.index,
    open = weekly_stock['Open'],
    high=weekly_stock['High'],
    low=weekly_stock['Low'],
    close=weekly_stock['Close'],
    increasing_line_color='#00FF00',
    decreasing_line_color='#FF0000',
    showlegend=False, 

    )

    trace2= go.Scatter(
        x = weekly_stock.index,
        y =weekly_stock['macd'],
        line=dict(color='#596E5C', width=2),
        name='macd'

    )

    trace3= go.Scatter(
        x = weekly_stock.index,
        y =weekly_stock['macd_h'],
        line=dict(color='#002408', width=2),
        name='macd high'

    )

    trace4= go.Scatter(
        x = weekly_stock.index,
        y =weekly_stock['macd_s'],
        line=dict(color='#B8293D', width=2),
        name='macd low'
    )

    trace5= go.Scatter(
        x = stock.index,
        y =stock['cum_gain'],
        line=dict(color='#B8293D', width=2),
        name='cummulative gains'
    )
    # figure= make_subplots(specs=[[{"secondary_y": True}]])
    figure= make_subplots(rows=4, cols=1)
    figure.add_trace(trace1, row=1, col = 1)
    # figure.add_trace(trace2,  row=3, col=1)
    figure.add_trace(trace3,  row=3, col=1)
    figure.add_trace(trace4,  row=3, col=1)
    figure.add_trace(trace5,  row=4, col=1)
    figure.update_layout(height=600)


    # price_chart = figure.to_html(full_html = False, include_plotlyjs=False)
    price_chart = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    trace1 =go.Candlestick(
        x=stock.index,
        open = stock['Open'],
        high=stock['High'],
        low=stock['Low'],
        close=stock['Close'],
        increasing_line_color='#00FF00',
        decreasing_line_color='#FF0000',
        showlegend=False
    )

    trace2= go.Scatter(
        x = stock.index,
        y =stock['macd'],
        line=dict(color='#596E5C', width=2),
        name='macd'

    )

    trace3= go.Scatter(
        x = stock.index,
        y =stock['macd_h'],
        line=dict(color='#002408', width=2),
        name='macd high'

    )

    trace4= go.Scatter(
        x = stock.index,
        y =stock['macd_s'],
        line=dict(color='#B8293D', width=2),
        name='macd low'

    )
    # figure= make_subplots(specs=[[{"secondary_y": True}]])
    figure= make_subplots(rows=3, cols=1)
    figure.add_trace(trace1, row=1, col = 1)
    # figure.add_trace(trace2,  row=3, col=1)
    figure.add_trace(trace3,  row=3, col=1)
    figure.add_trace(trace4,  row=3, col=1)
    figure.update_layout(height=1200)


    price_estimates = {
        "ticker" : stock_ticker,
        "open_price" : round(open_price[0],2),
        "low_estimate" : round(low_estimate[0],2),
        "high_estimate" :round(high_estimate[0],2),
        "current_price" : round(current_price[0],2),
        "current_low_estimate": round(current_low_estimate[0],2),
        "current_high_estimate": round(current_high_estimate[0],2)
    }
    daily_price_chart = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    final_result = (chart_result, confidence, price_chart, estimates, price_estimates, daily_price_chart)


    # print(chart_result)
    # return jsonify(final_result)
    return final_result