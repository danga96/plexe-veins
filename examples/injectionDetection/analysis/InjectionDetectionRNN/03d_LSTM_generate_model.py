import os
# Not show error message from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#disable GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from keras.callbacks import EarlyStopping
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
#os.system("taskset -p -c 0-15 %d" % os.getpid())
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

            #print("Shape X:",X.shape," X_train", self.X_train[name_value].shape, " ", self.X_train[name_value])

    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def train_model(self, name_value):
        print("--------------------------------------------",name_value,"-------------------------------------")
        X_train = self.X_train[name_value]
        y_train = self.y_train[name_value]

        scaler = StandardScaler()

        X_one_column = X_train.reshape([-1,1])
        result_one_column = scaler.fit_transform(X_one_column)
        X_train = (X_train - scaler.mean_)/(scaler.var_**0.5)

        f = open("./RollingDB/Model/scaler2_"+name_value[:-4]+".txt","w+")
        f.write(str(round(scaler.mean_[0],7))+" "+str(round(scaler.var_[0]**0.5,7)))
        f.close()

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        
        #print("shape_X",X_train.shape,"shape_Y",y_train.shape,"\n")
        
        model = Sequential()
        
        inputs = Input(shape=(10,1))
        L1r = GRU(128, activation='relu', return_sequences=True, kernel_initializer='uniform')(inputs)
        L1rd = Dropout(0.2)(L1r)
        L2r = GRU(128, activation='relu', return_sequences=False, kernel_initializer='uniform')(L1rd)
        L2rd = Dropout(0.2)(L2r)
        Fl = Flatten()(L2rd)
        L1 = Dense(256, activation='relu', kernel_initializer='uniform')(Fl)
        L1d = Dropout(0.2)(L1)
        L2 = Dense(128, activation='relu', kernel_initializer='uniform')(L1d)
        L2d = Dropout(0.2)(L2)
        L3 = Dense(64, activation='relu', kernel_initializer='uniform')(L2d)
        L3d = Dropout(0.2)(L3)
        L4 = Dense(32, activation='relu', kernel_initializer='uniform')(L3d)
        L4d = Dropout(0.2)(L4)
        predictions = Dense(1, activation='sigmoid', kernel_initializer='uniform')(L4d)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=30)
        model.fit(X_train,y_train, epochs=50, batch_size=32, verbose=1, callbacks=[es])

        joblib.dump(scaler,"./RollingDB/Model/scaler_"+name_value[:-4]+".bin",compress=True)
        model.save("./RollingDB/Model/model_"+name_value[:-4]+".h5", include_optimizer=False)

        return True


if __name__ == "__main__":
    train_path = "./RollingDB/"
    scenario = "Random" #Constant

    #NoAttack
    AllValues = ["KFdistance.csv",  "Rdistance.csv", "RKFdistance.csv", "RKFspeed.csv", "RV2Xspeed.csv", "V2XKFdistance.csv", "V2XKFspeed.csv","KFspeed.csv"]
    #AllValues = ["KFdistance.csv","V2XKFdistance.csv", "V2XKFspeed.csv"]
    #AllValues = ["RKFdistance.csv"]
    start_time = time.time()

    generator = GenerateModel(train_path, AllValues)

    num_workers = 8
    params = AllValues

    pool = multiprocessing.Pool(num_workers, generator.init_worker)

    scores = pool.map(generator.train_model, params)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
