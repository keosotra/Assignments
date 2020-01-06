
import pandas as pd
from collections import deque
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

test = yf.download("AAU, AEM, AG, AGI, ALO, ASA, ASM, AU, AUG, AUMN, AUY, AXU, BHP, BTG, BVN, CCJ, CDE, CHNR, CLF, CMCL, DNN, DRD, EGO, EQX, EXK, FCX, FNV, FSM, GDX, GDXJ, GFI, GLD, GMO, GOLD, GORO, GPL, GSS, HBM, HL, HMY, IAG, IWM, JNUG, KGC, KL, LAC, LODE, MAG, MMX, MPVD, MSB, MUX, NAK, NEM, NEXA, NG, NGD, NUGT, NXE, OPNT, OR, PAAS, PLG, PLM, PVG, PZG, QQQ, RGLD, RIO, SA, SAND, SBGL, SCCO, SIL, SILJ, SLV, SMTS, SPY, SSRM, SVM, TGB, THM, TMQ, TRQ, TRX, UEC, URG, USAS, USAU, VALE, VGZ, VTI, WPM, WRN, WWR, XPL", 
                   period='7d', interval='1m')

AccuracyList2 = []
count = 0
for i in range(10):

    SEQ_LEN = 30 # how long of a preceeding sequence to collect for RNN
    FUTURE_PERIOD_PREDICT = 15  # how far into the future are we trying to predict?
    RATIO_TO_PREDICT = "Close_GDX"
    EPOCHS = 10  # how many passes through our data
    BATCH_SIZE = 5  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
    def classify(current, future):
        if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
            return 1
        else:  # otherwise... it's a 0!
            return 0


    def preprocess_df(df, shuffle = False, balance = False):
        df = df.drop("future", 1)  # don't need this anymore.

    #     for col in df.columns:  # go through all of the columns
    #         if col != "target":  # normalize all ... except for the target itself!
    #             df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
    #             df.dropna(inplace=True)  # remove the nas created by pct_change
    #             df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

        targetlist = df['target'].tolist()
        df = df.drop("target", 1)
        df = pd.DataFrame(min_max.fit_transform(df), columns = df.columns)
        df['target'] = targetlist 
#         df.dropna(inplace=True)  # cleanup again... jic.


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




    main_df = test.copy()
    # main_df.dropna(inplace=True)
    # main_df = main_df.iloc[1:]
    main_df = main_df.round(3)
    main_df.index = main_df.index.astype(int)
    main_df.columns = main_df.columns.map('_'.join)
    # main_df = main_df.replace(to_replace=0, method='ffill')



    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    # main_df.dropna(inplace=True)
    main_df=main_df.mask(main_df==0).fillna(main_df.mean())

    main_df['future'] = main_df[f'{RATIO_TO_PREDICT}'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}'], main_df['future']))
    main_df.dropna(inplace=True)
    # main_df = main_df.reset_index(drop=True)
    # main_df.index = np.arange(1, len(main_df) + 1)
    main_df.index.name = 'time'
    # main_df.dropna(inplace=True)
    main_df.fillna(method="ffill", inplace=True) 
    len(main_df), len(test)


    # In[ ]:


    ## here, split away some slice of the future data from the main main_df.
    # times = sorted(main_df.index.values)
    # last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]
    # validation_main_df = main_df[(main_df.index >= last_5pct)]
    # main_df = main_df[(main_df.index < last_5pct)]
    # train_x, train_y = preprocess_df(main_df)
    # validation_x, validation_y = preprocess_df(validation_main_df)



    main_df1, validation_main_df, test_main_df = np.split(main_df, [int(.8*len(main_df)), int(.95*len(main_df))])
    train_x, train_y = preprocess_df(main_df1, shuffle=True, balance=True)
#     train_x, train_y = preprocess_df(main_df1)
    validation_x, validation_y = preprocess_df(validation_main_df, shuffle=True, balance=True)
    test_x, test_y = preprocess_df(test_main_df, balance=True)


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

    tensorboard = TensorBoard(log_dir="D:\\Python\\New folder\\logs",  profile_batch = 100000000)

    filepath = "D:\\Python\\New folder\\logs\\RNNV1_Final-{epoch:02d}-{val_accuracy:.2f}.h5"  # unique file name that will include the epoch and the validation acc for that epoch
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
    AccuracyList2.append(f'Iteration: {count} ACC: {score[1]} Loss: {score[0]} ||| ')
    count = count +1
    print(f'Iteration: {i}')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save(r"D:\Python\New folder\models\test.h5")
    

# from tensorflow import keras
# path = r'C:\Users\keosotra.veng\Desktop\Data\models'
# new_model = keras.models.load_model(path)

# score = new_model.evaluate(validation_x, np.asarray(validation_y), verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
