#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
from collections import deque
import tensorflow as tf
import pandas_ta as ta
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.callbacks import ReduceLROnPlateau
min_max = MinMaxScaler()
import yfinance as yf
pd.set_option('display.max_rows', 700)
pd.set_option('display.max_columns', 200)


# In[2]:


SEQ_LEN = 60 # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 30  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "close"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 5  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


# In[3]:


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


# In[4]:


def clean_data(df, for_train=False, target=False):
    df = df.round(2)
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    df = df.mask(df==0).fillna(method='bfill')
    df = df.mask(df==0).fillna(method='ffill')
    
    if target == True:
        df['future'] = df[f'{RATIO_TO_PREDICT}'].shift(-FUTURE_PERIOD_PREDICT)
        df['target'] = list(map(classify, df[f'{RATIO_TO_PREDICT}'], df['future']))
    
    if for_train==True:
        df.dropna(inplace=True)

    df.index.name = 'time'
    return df


# In[5]:


def preprocess_df(df, shuffle = False, balance = False, scale = False):
    df = df.drop("future", 1)  # don't need this anymore.

    if scale == True:
        targetlist = df['target'].tolist()
        df = df.drop("target", 1)
        df = pd.DataFrame(min_max.fit_transform(df), columns = df.columns)
        df['target'] = targetlist 
        df = df.round(2)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
    if shuffle == True:
        random.shuffle(sequential_data)  # shuffle for good measure.
        
    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!
            
    if shuffle == True:
        random.shuffle(buys)  # shuffle the buys
        random.shuffle(sells)  # shuffle the sells!

    if balance == True:
        lower = min(len(buys), len(sells))  # what's the shorter length?
        buys = buys[:lower]  # make sure both lists are only up to the shortest length.
        sells = sells[:lower]  # make sure both lists are only up to the shortest length.
        sequential_data = buys+sells  # add them together

    if shuffle == True:
        random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


# In[103]:


df = yf.download("GDX, GLD", 
                  start="2020-01-17", end="2020-01-20", interval='1m')
df.columns = df.columns.map('_'.join)
df.drop(df.tail(2).index,inplace=True) 
df = df.rename({'Close_GDX': 'close', 'High_GDX':'high', 'Low_GDX':'low'}, axis=1)  # new method
df = df[['Close_GLD', 'close', 'high', 'low', 'Volume_GDX', 'Volume_GLD']]


# In[104]:


main_df = df.copy()
main_df = clean_data(main_df, for_train=False, target=True)
# main_df.ta.macd(append=True)
# main_df.ta.sma(length= 60, append=True)

# main_df.ta.bbands(length = 15, append=True)
# main_df.ta.rsi(append=True)
# main_df.ta.adx(  append=True)
# main_df.ta.stoch(  append=True)

# main_df.dropna(inplace=True)
main_df


# In[97]:


len(main_df), len(df)


# In[105]:


test_x, test_y = preprocess_df(main_df, balance=False, shuffle=False, scale=True)


# In[92]:


test_x[0]


# In[106]:


from tensorflow import keras
path = r'C:\Users\keosotra.veng\Desktop\Data\models\test.h5'
new_model = keras.models.load_model(path)
score_classes = new_model.predict_classes(test_x)
score = new_model.evaluate(test_x, np.asarray(test_y), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[76]:


len(test_x)


# In[77]:


score_classes


# In[107]:


df2 = main_df[59:]
df2['predicted'] = score_classes
df2

