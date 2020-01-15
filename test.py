main_df = yf.download("GDX, GLD", 
                   period='60d', interval='5m')
main_df.columns = main_df.columns.map('_'.join)
main_df.drop(main_df.tail(2).index,inplace=True) 
main_df = main_df.rename({'Close_GDX': 'close'}, axis=1)  # new method
main_df = main_df[['Close_GLD', 'close', 'Volume_GDX', 'Volume_GLD']]
main_df = clean_data(main_df, for_train=True)
main_df.ta.macd(append=True)
main_df.dropna(inplace=True)

-----------------------

import pandas as pd
from collections import deque
import tensorflow as tf
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
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 200)


# In[284]:


SEQ_LEN = 30 # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 15  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "Close_GDX"
EPOCHS = 3  # how many passes through our data
BATCH_SIZE = 5  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


# In[285]:


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


# In[286]:


def clean_data(df, for_train=False):
    df = df.round(2)
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    df = df.mask(df==0).fillna(method='bfill')
    df = df.mask(df==0).fillna(method='ffill')
    
    df['future'] = df[f'{RATIO_TO_PREDICT}'].shift(-FUTURE_PERIOD_PREDICT)
    df['target'] = list(map(classify, df[f'{RATIO_TO_PREDICT}'], df['future']))
    
    if for_train==True:
        df.dropna(inplace=True)

    df.index.name = 'time'
    return df


# In[287]:


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


# In[288]:


main_df = yf.download("GDX, GLD", 
                   period='7d', interval='1m')
main_df.columns = main_df.columns.map('_'.join)
main_df.drop(main_df.tail(2).index,inplace=True) 
main_df = main_df[['Close_GLD', 'Close_GDX', 'Volume_GDX', 'Volume_GLD']]
main_df = clean_data(main_df, for_train=True)


# In[289]:


main_df1, validation_main_df, test_main_df = np.split(main_df, [int(.8*len(main_df)), int(.9*len(main_df))])
train_x, train_y = preprocess_df(main_df1, shuffle=True, balance=True, scale=True)
validation_x, validation_y = preprocess_df(validation_main_df, shuffle=True, balance=True, scale=True)
test_x, test_y = preprocess_df(test_main_df, balance=True, shuffle=False, scale=True)


# In[290]:


print(f"train data: {len(train_x)} validation: {len(validation_x)} test: {len(test_x)}")
print(f"Train Dont buys: {train_y.count(0)}, Train buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
print(f"Test Dont buys: {test_y.count(0)}, Test buys: {test_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0, nesterov=False)
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="C:\\Users\\keosotra.veng\\Desktop\\Data\\logs",  profile_batch = 100000000)

filepath = "C:\\Users\\keosotra.veng\\Desktop\\Data\\logs\\RNNV1_Final-{epoch:02d}-{val_accuracy:.2f}.h5"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') # saves only the best ones

# Train model
history = model.fit(
    train_x, np.asarray(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, np.asarray(validation_y)),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(test_x, np.asarray(test_y), verbose=0)
# AccuracyList2.append(f'Iteration: {count} ACC: {score[1]} Loss: {score[0]} ||| ')
# count = count +1
# print(f'Iteration: {i}')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save(r"C:\Users\keosotra.veng\Desktop\Data\models\test.h5")


# In[291]:


# main_df = yf.download("GDX, GLD", 
#                    period='7d', interval='1m')
# main_df.columns = main_df.columns.map('_'.join)
# # main_df.drop(main_df.tail(2).index,inplace=True) 
# main_df = main_df.tail(10)
# main_df = main_df[['Close_GLD', 'Close_GDX', 'Volume_GDX', 'Volume_GLD']]
# main_df = clean_data(main_df, for_train=False)


# In[292]:


main_df


# In[293]:


# test_no_bal_x, test_no_bal_y = preprocess_df(main_df, balance=False, scale=False)
# test_no_bal_x


# In[294]:


test_no_bal_y


# In[295]:


# test_x, test_y = preprocess_df(main_df, balance=True, scale=False)
# test_x


# In[296]:


test_y

