import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dropout
from keras.layers import Input, Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers
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
from multiprocessing.pool import ThreadPool as Pool

# fix random seed for reproducibility
tf.random.set_seed(2)
np.random.seed(7)
#os.system("taskset -p -c 0 %d" % os.getpid())
class GenerateModel:
    def __init__(self, base_path, AllValues):
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}

        for name_value in AllValues:
            #print("--------------------------------------------",name_value,"-------------------------------------")
            value_data = pd.read_csv(base_path+name_value)
            #value_data = shuffle(value_data)
            X = value_data.drop(['Detection'], axis=1).values
            Y = value_data["Detection"].values

            self.X_train[name_value] = X
            self.y_train[name_value] = Y
            #self.X_train[name_value], self.X_test[name_value], \
            #    self.y_train[name_value], self.y_test[name_value] = train_test_split(X, Y, test_size = 0.01, shuffle=True)

            #print("Shape X:",X.shape," X_train", self.X_train[name_value].shape, " ", self.X_train[name_value])

    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def train_model(self, name_value):
        print("--------------------------------------------",name_value,"-------------------------------------")
        X_train = self.X_train[name_value]
        #X_test = self.X_test[name_value]
        y_train = self.y_train[name_value]
        #y_test = self.y_test[name_value]

        scaler = StandardScaler()
        #print("X_Train_10", X_train[10])
        X_one_column = X_train.reshape([-1,1])
        result_one_column = scaler.fit_transform(X_one_column)
        X_train = (X_train - scaler.mean_)/(scaler.var_**0.5)

        f = open("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/scaler2_"+name_value[:-4]+".txt","w+")
        f.write(str(round(scaler.mean_[0],7))+" "+str(round(scaler.var_[0]**0.5,7)))
        #f.writelines([scaler.mean_+", ", scaler.var_]) 
        f.close()


        #X_train = result_one_column.reshape(X_train.shape)
        #X_train = scaler.fit_transform(X_train)
        #X_test = scaler.transform(X_test)
        
        #print("X_Train_10", X_train[10])
        #print("Mean scaler: ", scaler.mean_)
        #print("Std scaler: ", scaler.var_**0.5)
        #joblib.dump(scaler,"/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/scaler_"+name_value[:-4]+".bin",compress=True)
        #exit()
        #return True

        
        
        ##X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        
        #print("shape_X",X_train.shape,"shape_Y",y_train.shape,"\n")
        
        model = Sequential()
        """
        model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(64, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        """
        #model.add(Conv1D(filters=128, kernel_size=5, activation='tanh', input_shape=(10,1)))
        #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        #model.add(Flatten())
        #model.add(GRU(128, input_shape=(X_train.shape[1:]), activation='tanh', return_sequences=True, kernel_initializer='glorot_uniform'))
        #model.add(Dropout(0.4))
        #model.add(GRU(256, input_shape=(X_train.shape[1:]), activation='tanh', kernel_initializer='glorot_uniform',return_sequences=True))
        #model.add(GRU(50, activation='tanh', return_sequences=True, kernel_initializer='glorot_uniform'))
        #model.add(Dropout(0.4))
        #model.add(Dropout(0.6))
        #model.add(RepeatVector(n=X_train.shape[1]))
        #model.add(GRU(64, activation='relu',return_sequences=True))
        #model.add(Dropout(0.1))
        #model.add(LSTM(64, activation='relu', return_sequences=False))
        #model.add(Dropout(0.2))
        #model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        #model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(10,1)))
        #model.add(MaxPooling1D(pool_size=3))
        #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        #model.add(MaxPooling1D(pool_size=2,strides=1))
        #model.add(Dropout(0.1))

        #model.add(LSTM(128, activation='tanh', return_sequences=False, kernel_initializer='glorot_uniform'))
        #model.add(Flatten())
        
        #model.add(Dropout(0.4))
        #model.add(GRU(20, activation='tanh', return_sequences=False, kernel_initializer='glorot_uniform'))
        #model.add(Dropout(0.6))
        #model.add(Dense(256, activation='relu', input_dim=10 , kernel_initializer='glorot_uniform'))
        #model.add(Flatten())
        """
        model.add(Dense(256, activation='tanh', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation='tanh', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation='tanh', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.1))
        
        model.add(Dense(64, activation='tanh',  kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.1))
        """
        #model.add(Flatten())
        #model.add(Dense(128, activation='tanh',  kernel_initializer='uniform'))
        #model.add(Dropout(0.1))
        #model.add(Dense(64, activation='tanh',  kernel_initializer='glorot_uniform'))
        #model.add(Dropout(0.2))
        #model.add(Dense(64, activation='tanh',  kernel_initializer='uniform'))
        
        #model.add(Dense(32, activation='tanh', kernel_initializer='uniform'))
        
        #model.add(TimeDistributed(Dense(1)))
        """
        model.add(Dense(1, activation='sigmoid',  kernel_initializer='glorot_uniform'))
        opt = optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
        #print(model.summary())
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
        """

        """
        #----------------------CONF 1------------------------------
        model.add(GRU(128, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=False, kernel_initializer='glorot_uniform'))
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
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)   
        #----------------------END CONF 1----------------------
        """
        """
        model.add(GRU(64, activation='relu', return_sequences=True, kernel_initializer='uniform'))		
        model.add(Dropout(0.2))
        model.add(GRU(32, activation='relu', return_sequences=False, kernel_initializer='uniform'))		
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        """
        
        
        inputs = Input(shape=(10,1))
        L1r = GRU(64, activation='relu', return_sequences=True, kernel_initializer='uniform')(inputs)
        L1rd = Dropout(0.2)(L1r)
        L2r = GRU(32, activation='relu', return_sequences=False, kernel_initializer='uniform')(L1rd)
        L2rd = Dropout(0.2)(L2r)
        Fl = Flatten()(L2rd)
        L1 = Dense(32, activation='relu', kernel_initializer='uniform')(Fl)
        predictions = Dense(1, activation='sigmoid', kernel_initializer='uniform')(L1)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train,y_train, epochs=5, batch_size=32)
        



        # Final evaluation of the model
        """
        score = model.evaluate(X_test, y_test, verbose=0)


        Y_pred = model.predict(X_test)
        Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
        confusion_matrix = metrics.confusion_matrix(y_test, Y_pred_round)
        scores = {
            'name' : name_value,
            'score' : score,
            'matrix':confusion_matrix
        }
        """
        joblib.dump(scaler,"/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/scaler_"+name_value[:-4]+".bin",compress=True)
        model.save("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_"+name_value[:-4]+".h5", include_optimizer=False)
        #model.save("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_"+name_value[:-4]+".h5")

        #return scores
        return True


if __name__ == "__main__":
    train_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/"
    scenario = "Random" #Constant

    #NoAttack
    AllValues = ["KFdistance.csv",  "Rdistance.csv", "RKFdistance.csv", "RKFspeed.csv", "RV2Xspeed.csv", "V2XKFdistance.csv", "V2XKFspeed.csv","KFspeed.csv"]
    #AllValues = ["KFdistance.csv","V2XKFdistance.csv", "V2XKFspeed.csv"]
    AllValues = ["V2XKFspeed.csv"]
    start_time = time.time()
    #AllAttacks = ["{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    generator = GenerateModel(train_path, AllValues)

    num_workers = 1
    params = AllValues

    pool = multiprocessing.Pool(num_workers, generator.init_worker)
    #pool = Pool(num_workers)
    scores = pool.map(generator.train_model, params)
    """
    for score in scores:
        print("------------------",score['name'],"----------------------")
        print(score['matrix'])
        print(score['score'])
    """
    #print(scores)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  