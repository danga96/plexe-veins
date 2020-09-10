import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf    
import time
import signal
import multiprocessing

# fix random seed for reproducibility
tf.random.set_seed(2)
np.random.seed(7)

class GenerateModel:
    def __init__(self, base_path, AllValues):
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}

        for name_value in AllValues:
            #print("--------------------------------------------",name_value,"-------------------------------------")
            value_data = pd.read_csv(base_path+name_value)
            X = value_data.drop(['Detection'], axis=1).values
            Y = value_data["Detection"].values

            self.X_train[name_value], self.X_test[name_value], \
                self.y_train[name_value], self.y_test[name_value] = train_test_split(X, Y, test_size = 0.01, shuffle=True)

            #print("Shape X:",X.shape," X_train", self.X_train[name_value].shape, " ", self.X_train[name_value])

    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def train_model(self, name_value):
        print("--------------------------------------------",name_value,"-------------------------------------")
        X_train = self.X_train[name_value]
        X_test = self.X_test[name_value]
        y_train = self.y_train[name_value]
        y_test = self.y_test[name_value]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        
        
        
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        
        #print("shape_X",X_train.shape,"shape_Y",y_train.shape)

        model = Sequential()
        """
        model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(64, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        """
        model.add(GRU(128, input_shape=(X_train.shape[1:]), activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(512, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(256, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(64, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
        #model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print(model.summary())
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=0)


        Y_pred = model.predict(X_test)
        Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
        confusion_matrix = metrics.confusion_matrix(y_test, Y_pred_round)
        scores = {
            'name' : name_value,
            'score' : score,
            'matrix':confusion_matrix
        }
        
        joblib.dump(scaler,"/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/scaler_"+name_value[:-4]+".bin",compress=True)
        model.save("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_"+name_value[:-4]+".h5")

        return scores


if __name__ == "__main__":
    train_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/"
    scenario = "Random" #Constant

    #NoAttack
    AllValues = ["KFdistance.csv",  "Rdistance.csv", "RKFdistance.csv", "RKFspeed.csv", "RV2Xspeed.csv", "V2XKFdistance.csv", "V2XKFspeed.csv",]
    #AllValues = ["KFdistance.csv",  "Rdistance.csv"]
    start_time = time.time()
    #AllAttacks = ["{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    generator = GenerateModel(train_path, AllValues)

    num_workers = 1
    params = AllValues

    pool = multiprocessing.Pool(num_workers, generator.init_worker)

    scores = pool.map(generator.train_model, params)

    for score in scores:
        print("------------------",score['name'],"----------------------")
        print(score['matrix'])
        print(score['score'])

    #print(scores)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  