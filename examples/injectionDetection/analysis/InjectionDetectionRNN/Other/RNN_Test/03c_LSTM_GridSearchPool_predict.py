import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle
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
from keras import optimizers
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

#os.system("taskset -p -c 0 %d" % os.getpid())

class FindModel:
    def __init__(self, base_path, name_value):

        #print("--------------------------------------------",name_value,"-------------------------------------")
        value_data = pd.read_csv(base_path+name_value)
        value_data = shuffle(value_data)
        X = value_data.drop(['Detection'], axis=1).values
        Y = value_data["Detection"].values
        X_train = X
        self.y_train = Y
        #X_train, X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        #X_test = scaler.transform(X_test)

        
        #self.X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

        self.X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        #print("Shape X:",X.shape," X_train", self.X_train[name_value].shape, " ", self.X_train[name_value])

    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _get_optimizer(self, _optimizer, learn_rate):
        #optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        switcher = {
            'SGD': optimizers.SGD(learning_rate=learn_rate),
            'RMSprop': optimizers.RMSprop(learning_rate=learn_rate),
            'Adagrad': optimizers.Adagrad(learning_rate=learn_rate),
            'Adadelta': optimizers.Adadelta(learning_rate=learn_rate),
            'Adam': optimizers.Adam(learning_rate=learn_rate),
            'Adamax': optimizers.Adamax(learning_rate=learn_rate),
            'Nadam': optimizers.Nadam(learning_rate=learn_rate)
        }
        return switcher.get(_optimizer, 0)

    def train_model(self, conf):
        """
        conf: [[lstm_layer], epochs, batch_size, activation, learn_rate, dropout_rate, init, optimizer]
                    0           1           2           3           4           5         6       7
        """
        print("-------------",conf,"-----")
        lstm_layer=0 
        """
        lstm_layer=0 
        dropout_rate=0.0
        activation = 'relu'
        optimizer='Adam'
        learn_rate=0.001
        init_mode='glorot_uniform'
        """
        X_train = self.X_train
        #X_test = self.X_test
        y_train = self.y_train
        #y_test = self.y_test

        lstm_layer = list(map(int,conf[0].split(',')))
        epochs = int(conf[1])
        batch_size = int(conf[2])
        activation = conf[3]
        learn_rate = float(conf[4])
        dropout_rate = float(conf[5])
        init_mode = conf[6]
        optimizer = conf[7]
        #print("\n",list(map(int,test[0][0].split(','))))
 
        #print(dense_layer, epochs, batch_size, activation, learn_rate, dropout_rate, init_mode, optimizer)      
        
        #exit()
        #print("shape_X",X_train.shape,"shape_Y",y_train.shape)

        first_layer_size_LSTM= 16 if lstm_layer==0 else lstm_layer[0]
        last_layer_size_LSTM = 16 if lstm_layer==0 else lstm_layer[-1]
        model = Sequential()
        model.add(GRU(first_layer_size_LSTM, input_shape=(10,1), kernel_initializer=init_mode, activation=activation, return_sequences=True if len(lstm_layer)>1 else False))
        model.add(Dropout(dropout_rate))

        #print("FISR:",first_layer_size_LSTM," LASST:",last_layer_size_LSTM)

        for layer_size in lstm_layer[1:-1]:
            #print("IIIIIIIIIIIIINNNNNNNNNNNNNNNNN           FFFFFFFFFFFOOOOOOOOOOOORRRRRRRRRRR")
            model.add(GRU(layer_size, kernel_initializer=init_mode, activation= activation, return_sequences=True))
            model.add(Dropout(dropout_rate))

        if len(lstm_layer) >= 2 :
            model.add(GRU(last_layer_size_LSTM, kernel_initializer=init_mode, activation= activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(64, kernel_initializer=init_mode, activation= activation))
        model.add(Dropout(dropout_rate))

        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        _optimizer = self._get_optimizer(optimizer, learn_rate)
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
        #print(model.summary())
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Final evaluation of the model
        #score = model.evaluate(X_test, y_test, verbose=0)


        #Y_pred = model.predict(X_test)
        #Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
        #confusion_matrix = metrics.confusion_matrix(y_test, Y_pred_round)
        scores = {
            'conf' : conf,
            'accuracy' : history.history['accuracy'],
            'loss' : history.history['loss']
        }
        
        #model.save("/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/model_"+name_value[:-4]+".h5")

        return scores


if __name__ == "__main__":
    train_path = "../../RollingDB/"
    scenario = "Random" #Constant

    #NoAttack
    #AllValues = ["KFdistance.csv",  "Rdistance.csv", "RKFdistance.csv", "RKFspeed.csv", "RV2Xspeed.csv", "V2XKFdistance.csv", "V2XKFspeed.csv",]
    name_value = "V2XKFdistance.csv"
    start_time = time.time()
    #AllAttacks = ["{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    generator = FindModel(train_path, name_value)


    ####################################################################
    #configurations = [ [[256,128,64],0.2], [[100,50],0.5] ]
    lstm_size_candidates = ['256,128,64', '32', '512,256,128,64', '256,128,128,64']
    epochs = [100]
    batch_size = [16,32,64,512]   
    # Use scikit-learn to grid search 
    #activation =  ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax', 'softplus', 'softsign']
    activation =  ['relu', 'tanh', 'sigmoid', 'softsign']
    #learn_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    learn_rate = [0.001, 0.01]
    #dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dropout_rate = [0.0, 0.1, 0.2, 0.5, 0.7, 0.8]
    #dropout_rate = [0.2,0.0]
    #init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    init = ['uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizer = [ 'SGD', 'Adam', 'Adamax']
    
    #########################(START) test one_by_one###########################################
    #dense_size_candidates = ['256,128,64', '32', '512,256,128,64', '256,128,128,64']
    lstm_size_candidates = ['128']
    epochs = [1]
    batch_size = [32]    
    # Use scikit-learn to grid search 
    activation =  ['relu']
    learn_rate = [0.001]
    dropout_rate = [0.1]
    init = ['uniform']
    optimizer = ['Adam']
    #########################(END) test one_by_one#############################################
    


    configurations = np.array(np.meshgrid(lstm_size_candidates,epochs,batch_size,activation,learn_rate,dropout_rate,init,optimizer)).T.reshape(-1,8)
    print("CONFIGURATIONS: ",configurations,"\n",configurations.shape)
    time.sleep(2)

    num_workers = 1
    params = configurations

    pool = multiprocessing.Pool(num_workers, generator.init_worker)

    scores = pool.map(generator.train_model, params)

    #########################FIND BEST CONFIGURATIONS#############################################

    for score in scores:
        print("------------------",score['conf'],"------------------")
        print("------------------",score['accuracy'],"------------------",score['loss'])
       

    #print(scores)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  