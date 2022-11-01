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

def get_options_date(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    return stock_data.options
