
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns


os.environ["KERAS_BACKEND"] = "tensorflow"


import yfinance as yf

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
                hstprice['Datetime'] = pd.to_datetime(hstprice['Datetime'])
                hstprice['date'] = hstprice['Datetime'].dt.date
                hstprice['time'] = hstprice['Datetime'].dt.time
                hstprice['open_close_hr'] = hstprice['time'].astype(str).apply(lambda x: 1 if x == '09:30:00' else (2 if x == '15:30:00' else 0))
                hstprice['mov_avg'] = hstprice['Mean'].rolling(window=50).mean()
                hstprice['mov_avg_diff'] = (hstprice['mov_avg']/hstprice['Mean'])-1
                hstprice['inside_days'] = hstprice.apply(lambda x: 1 if (x['Open'] < x['prev_open']) and (x['Close'] < x['prev_close']) else 0, axis=1)
                hstprice['vol_var'] = (hstprice['Volume']/(hstprice['Volume'].median()))-1
                hstprice['abs_mnChng'] = abs(hstprice['mean_change'])
                hstprice['channel_ptrn'] = hstprice['abs_mnChng'].rolling(window=12).mean()
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

    def scaler(self,target_col,time_col=None):
        df = self.df.copy(deep=True)
        features = df.drop(columns=[target_col])
        target = df[target_col]
        if time_col is not None:
            features = ml_preprocess.preprocess_time(features, time_col)
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        return features, target


    

class nn_model:

    def __init__(self):
        pass

    def feedforward_construct(self):
        model = Sequential()
        model.add(layers.Dense(input_dim=len(X_train.columns),units=10,activation='relu'))
        model.add(layers.Dense(input_dim=10,units=1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.01),metrics=['accuracy'])
        fitted_model = model.fit(X_train, y_train, epochs=15, batch_size=32)
        return fitted_model

    def model_accuracy(self):
        pass

    def tt_eval(self):
        y_train_pred_proba = model.predict(X_train)
        y_test_pred_proba = model.predict(X_test)
        train_preds_df = pd.DataFrame(y_train_pred_proba, columns=['Predicted'], index=X_train.index)
        train_preds_df['True Label'] = y_train
        test_preds_df = pd.DataFrame(y_test_pred_proba, columns=['Predicted'], index=X_test.index)
        test_preds_df['True Label'] = y_test
        full_predictions_df = pd.concat([train_preds_df, test_preds_df])
        full_predictions_df = full_predictions_df.sort_index().reset_index()

        fp0 = full_predictions_df[full_predictions_df['True Label'] == 0]
        fp1 = full_predictions_df[full_predictions_df['True Label'] == 1]


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.kdeplot(x=fp0['index'], y=fp0['Predicted'], cmap='coolwarm', fill=True, alpha=0.5, ax=ax1)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.set_title('Predictions of True Classification 0')

        sns.kdeplot(x=fp1['index'], y=fp1['Predicted'], cmap='coolwarm', fill=True, alpha=0.5, ax=ax2)
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.set_title('Predictions of True Classification 1')

        plt.tight_layout()

        plt.show()






##################
# DATA CLEANING
##################

# Adding the data to the dataframe(s), handling missing data/data errors, making conversions, etc.

ticker = 'SPY'
start_date = '2023-01-01'
end_date = '2024-02-01'
sk = finHelp(tkr='SPY',type='historical_price',sdate=start_date,edate=end_date,intv='1h')
dv = deriveVar(tkr='SPY', type='historical_price', sdate=start_date, edate=end_date, intv='1h')
stk3 = sk.stockdf(price='all')

rsi_df = dv.rsi(period=13).drop(columns='Close')

sDf2 = stk3.merge(rsi_df,on='Datetime',how='left')
target_level = dv.targetvar(sDf2,'mean_change')
sDf2['target'] = sDf2['abs_mnChng'].apply(lambda x: 1 if x >= target_level else 0)
sDf2['adjusted_target'] = sDf2['target'].shift(-10)
sDf2 = sDf2.drop(columns=['Open','Close','High','Low','Mean','gaps','mean_shift','mean_change',
                          'prev_open','prev_close','Datetime','date','mov_avg','abs_mnChng','target'])
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

hidden_units = 10
activation = 'relu'
l2 = 0.01
lr = 0.01
epochs = 5
batch_size = 16


model = Sequential()

model.add(layers.Dense(input_dim=len(X_train.columns),
                       units=hidden_units,
                       activation=activation))

model.add(layers.Dense(input_dim=hidden_units,
                       units=1,
                       activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=lr),
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=15, batch_size=batch_size)

train_acc = model.evaluate(X_train, y_train, batch_size=32)[1]
test_acc = model.evaluate(X_test, y_test, batch_size=32)[1]
print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)





##################
# PATTERN EVALUATION
##################

# Outcomes of cluster analysis (technically falls before the model training)
# Test scores of the NN on historical market data 
# Visual check of classification score to price movement 



y_train_pred_proba = model.predict(X_train)
y_train_pred = (y_train_pred_proba > 0.5).astype(int)


y_test_pred_proba = model.predict(X_test)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)


train_preds_df = pd.DataFrame(y_train_pred, columns=['Predicted'], index=X_train.index)
train_preds_df['True Label'] = y_train

test_preds_df = pd.DataFrame(y_test_pred, columns=['Predicted'], index=X_test.index)
test_preds_df['True Label'] = y_test

full_predictions_df = pd.concat([train_preds_df, test_preds_df])

full_predictions_df = full_predictions_df.sort_index()

full_dataset = sDf2.copy()
full_dataset = full_dataset.rename_axis('Index').reset_index()

merged_df = full_dataset.merge(full_predictions_df, left_on='Index', right_index=True, how='left')

merged_df








##################
# KNOWLEDGE REPRESENTATION
##################

# Interactive visualization of daily market behaviors to the classification score plus a running table of data 


ticker = 'SPY'
start_date = '2024-05-01'
end_date = '2024-08-01'
sk = finHelp(tkr='SPY',type='historical_price',sdate=start_date,edate=end_date,intv='1h')
dv = deriveVar(tkr='SPY', type='historical_price', sdate=start_date, edate=end_date, intv='1h')
stk4 = sk.stockdf(price='all')

rsi_df2 = dv.rsi(period=13).drop(columns='Close')

sDf3 = stk4.merge(rsi_df,on='Datetime',how='left')
sDf3['mov_avg'] = sDf3['Mean'].rolling(window=50).mean()
sDf3['mov_avg_diff'] = (sDf3['mov_avg']/sDf3['Mean'])-1
sDf3['inside_days'] = sDf3.apply(lambda x: 1 if (x['Open'] < x['prev_open']) and (x['Close'] < x['prev_close']) else 0, axis=1)
sDf3['vol_var'] = (sDf3['Volume']/(sDf3['Volume'].median()))-1
sDf3['abs_mnChng'] = abs(sDf3['mean_change'])
target_level = dv.targetvar(sDf3,'mean_change')
sDf3['target'] = sDf3['abs_mnChng'].apply(lambda x: 1 if x >= target_level else 0)
sDf3['adjusted_target'] = sDf3['target'].shift(-10)
sDf3['channel_ptrn'] = sDf3['abs_mnChng'].rolling(window=12).mean()
base_future = sDf3.copy(deep=True)
sDf3 = sDf3.drop(columns=['Open','Close','High','Low','Mean','gaps','mean_shift','mean_change','prev_open','prev_close','Datetime','date','mov_avg','abs_mnChng','target'])
sDf3 = sDf3.iloc[50:-10]


ml = ml_preprocess(df=sDf3)

new_features, new_target = ml.scaler(target_col='adjusted_target',time_col='time')

y_new_pred_proba = model.predict(new_features)
y_new_pred = (y_new_pred_proba > 0.5).astype(int)

#Consider changing the "new_features" with "base_future"
new_preds_df = pd.DataFrame(y_new_pred, columns=['Predicted'], index=new_features.index)
new_preds_df['True Label'] = new_target

accuracy = accuracy_score(new_target, y_new_pred)
print('Accuracy on new dataset:', accuracy)

#now just add on the base_future dataset



new_preds_df
