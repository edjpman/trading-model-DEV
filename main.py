
###### This the the main python file to run the mining process ######
# Additional testing will likely be completed in a Jupyter notebook testing environment 
# Official training of the model may be completed on a cloud platform for additional compute power 


##################
# REQUIREMENTS
##################

# Includes python libraries and the importing of helper functions 
# Importing of data/api calls 

import yfinance as yf
import pandas as pd
import numpy as np 
import tensorflow as tf 
import keras 
from keras import layers 


class finHelp:

    def __init__(self,tkr,type,sdate,edate,intv):
        self.tkr = tkr
        self.type = type 
        self.sdate = sdate
        self.edate = edate
        self.intv = intv

    def stockdf(self,price):
        tkr_fetch = yf.Ticker(self.tkr)

        if type == 'historical_price':
            hstprice = tkr_fetch.history(start=self.sdate,end=self.edate,interval=self.intv)

            if price == 'close':
                hstprice = hstprice['Close']
                return hstprice
            elif price == 'open':
                hstprice = hstprice['Open']
                return hstprice
            elif price == 'high':
                hstprice = hstprice['High']
                return hstprice
            elif price == 'low':
                hstprice = hstprice['Low']
                return hstprice
            elif price == 'mean':
                mnprice = (hstprice['High']+hstprice['Low'])/2
                hstprice['Mean'] = mnprice
                return hstprice
            else:
                return ValueError('Must pass "close" or "open"')

        elif type == 'volume':
            hstvol = tkr_fetch.history(start=self.sdate,end=self.edate,interval=self.intv)
            hstvol = hstvol['Volume']
            return hstvol
        
        else:
            return ValueError('Must pass "historical_price" or "volume"')


class deriveVar(finHelp):
    
    def __init__(self,tkr,type,sdate,edate,intv):
        super().__init__(tkr,type,sdate,edate,intv)
    
    def rsi(self,period):
        closeprc = self.stockdf('historical_price',start=self.sdate,end=self.edate,interval=self.intv,price='close')
        delta = closeprc.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rels = avg_gain/avg_loss

        rsi = 100 - (100/(1+rels))

        rsi_df = pd.DataFrame({
            'Close': closeprc,
            'RSI': rsi
        })

        return rsi_df







##################
# DATA CLEANING
##################

# Adding the data to the dataframe(s), handling missing data/data errors, making conversions, etc.



##################
# DATA INTEGRATION
##################

# Merging the clean data from the various sources for into a flat format for additional preparation for mining



##################
# DATA SELECTION
##################

# Parsing out variables that are not necessary for the inital stage of the mining 



##################
# DATA TRANSFORMATION
##################

# Deriving variables if needed, normalizing variables, creating the lagging scale for the classification targets 



##################
# DATA MINING
##################

# Feed-forward neural network development and training 



##################
# PATTERN EVALUATION
##################

# Outcomes of cluster analysis (technically falls before the model training)
# Test scores of the NN on historical market data 
# Visual check of classification score to price movement 



##################
# KNOWLEDGE REPRESENTATION
##################

# Interactive visualization of daily market behaviors to the classification score plus a running table of data 

