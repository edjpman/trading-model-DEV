
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
import os 
from keras import layers 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "tensorflow"


class finHelp:

    def __init__(self,tkr,type,sdate,edate,intv):
        self.tkr = tkr
        self.type = type 
        self.sdate = sdate
        self.edate = edate
        self.intv = intv

    def stockdf(self,price):
        tkr_fetch = yf.Ticker(self.tkr)

        if self.type == 'historical_price':
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
            elif price == 'all':
                hstprice = hstprice[['Open','Close','High','Low','Volume']].reset_index()
                hstprice['Mean'] = (hstprice['High']+hstprice['Low'])/2
                hstprice['gaps'] = hstprice['Close'].shift(1)
                hstprice['mean_shift'] = hstprice['Mean'].shift(1)
                hstprice['open_day_change'] = (hstprice['Open']/hstprice['gaps'])-1
                hstprice['mean_change'] = (hstprice['Mean']/hstprice['mean_shift'])-1
                hstprice['prev_open'] = hstprice['Open'].shift(1)
                hstprice['prev_close'] = hstprice['Close'].shift(1)
                # hstprice = hstprice.drop(columns=['Open','Low','High','Close','Mean','gaps','mean_shift'])
                hstprice['Datetime'] = pd.to_datetime(hstprice['Datetime'])
                hstprice['date'] = hstprice['Datetime'].dt.date
                hstprice['time'] = hstprice['Datetime'].dt.time
                hstprice['open_close_hr'] = hstprice['time'].astype(str).apply(lambda x: 1 if x == '09:30:00' else (2 if x == '15:30:00' else 0))
                return hstprice
            elif price == 'mean':
                mnprice = (hstprice['High']+hstprice['Low'])/2
                hstprice['Mean'] = mnprice
                return hstprice
            else:
                return ValueError('Must pass "close" or "open"')

        elif self.type == 'volume':
            hstvol = tkr_fetch.history(start=self.sdate,end=self.edate,interval=self.intv)
            hstvol = hstvol['Volume']
            return hstvol
        
        else:
            return ValueError('Must pass "historical_price" or "volume"')


class deriveVar(finHelp):
    
    def __init__(self,tkr,type,sdate,edate,intv):
        super().__init__(tkr,type,sdate,edate,intv)
    
    def rsi(self,period):
        closeprc = self.stockdf(price='close')
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

        rsi_df = rsi_df.reset_index()

        return rsi_df
    
    def targetvar(self,df,col):
        std_dev = df[col].std()
        return std_dev



class ml_preprocess:

    def __init__(self,df):
        self.df = df

    @staticmethod
    def preprocess_time(df, time_col):
        df = df.copy(deep=True)
        df[time_col] = pd.to_datetime(df[time_col], format='%H:%M:%S').dt.time
        df['hour'] = df[time_col].apply(lambda x: x.hour + x.minute / 60)
        df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df.drop(columns=[time_col, 'hour'], inplace=True)
        return df



    def tt_split(self,target_col,time_col=None):
        df = self.df.copy(deep=True)
        features = df.drop(columns=[target_col])
        target = df[target_col]
        if time_col is not None:
            features = ml_preprocess.preprocess_time(features, time_col)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2024)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        return X_train, X_test, y_train, y_test
    
    
    def batch_ds(self,traindf,testdf,val):
        trds = traindf.batch(val)
        tsds = testdf.batch(val)
        return trds, tsds
    
    

class nn_model:

    def __init__(self,l1_neurons,l2_neurons,l1_act,l2_act,out_neurons,out_act):
        l1_neurons.self = l1_neurons
        l2_neurons.self = l2_neurons
        l1_act.self = l1_act
        l2_act.self = l2_act
        out_neurons.self = out_neurons
        out_act.self = out_act

    def feedforward_construct(self):
        model = keras.Sequential()
        model.add(layers.Dense(self.l1_neurons, activation=self.l1_act))
        model.add(layers.Dense(self.l2_neurons, activation=self.l2_act))
        model.add(layers.Dense(4, activation=self.out_act))
        return model 
        
    def model_compile(self):
        pass

    def model_train(self):
        pass

    def model_test(self):
        pass







##################
# DATA CLEANING
##################

# Adding the data to the dataframe(s), handling missing data/data errors, making conversions, etc.

ticker = 'SPY'
start_date = '2023-01-01'
end_date = '2024-08-02'
sk = finHelp(tkr='SPY',type='historical_price',sdate=start_date,edate=end_date,intv='1h')
dv = deriveVar(tkr='SPY', type='historical_price', sdate=start_date, edate=end_date, intv='1h')
stk3 = sk.stockdf(price='all')

rsi_df = dv.rsi(period=13).drop(columns='Close')

sDf2 = stk3.merge(rsi_df,on='Datetime',how='left')
sDf2['mov_avg'] = sDf2['Mean'].rolling(window=50).mean()
sDf2['mov_avg_diff'] = (sDf2['mov_avg']/sDf2['Mean'])-1
sDf2['inside_days'] = sDf2.apply(lambda x: 1 if (x['Open'] < x['prev_open']) and (x['Close'] < x['prev_close']) else 0, axis=1)
#Need to confirm if it should be on a total volume avg or a rolling value
sDf2['vol_var'] = (sDf2['Volume']/(sDf2['Volume'].median()))-1
sDf2['abs_mnChng'] = abs(sDf2['mean_change'])
target_level = dv.targetvar(sDf2,'mean_change')
sDf2['target'] = sDf2['abs_mnChng'].apply(lambda x: 1 if x >= target_level else 0)
sDf2['adjusted_target'] = sDf2['target'].shift(-10)
#Few Issues: The model may place too much weight on this, could be highly correlated with target variable, may not work properly with the shifting of the target variable
sDf2['channel_ptrn'] = sDf2['abs_mnChng'].rolling(window=12).mean()
sDf2 = sDf2.drop(columns=['Open','Close','High','Low','Mean','gaps','mean_shift','mean_change','prev_open','prev_close','Datetime','date','mov_avg','abs_mnChng','target'])
sDf2 = sDf2.iloc[50:-10]


ml = ml_preprocess(df=sDf2)

X_train, X_test, y_train, y_test = ml.tt_split(target_col='adjusted_target',time_col='time')
print(X_train)




##################
# DATA INTEGRATION
##################

# Merging the clean data from the various sources for into a flat format for additional preparation for mining



##################
# DATA SELECTION
##################

# Parsing out variables that are not necessary for the inital stage of the mining 
# Splitting the dataset into test/train sets



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

