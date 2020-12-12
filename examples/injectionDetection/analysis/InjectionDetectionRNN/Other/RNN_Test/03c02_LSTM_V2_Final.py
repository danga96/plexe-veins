import os, glob
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
from keras.models import load_model
import tensorflow as tf    
import time
import signal
import multiprocessing
#import psutil

"""
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
"""

# fix random seed for reproducibility
tf.random.set_seed(2)
np.random.seed(7)
X_train = {}
y_train = {}
#os.system("taskset -p -c 0-15 %d" % os.getpid())
def get_XYscaler_Train(train_path,name_value):
    global X_train, y_train
    #print("--------------------------------------------",name_value,"-------------------------------------")
    value_data = pd.read_csv(train_path+name_value)
    #value_data = shuffle(value_data, random_state = 7)
    X = value_data.drop(['Detection'], axis=1).values
    Y = value_data["Detection"].values
    X_train = X
    y_train = Y
    name_value = name_value[:-4]

    scaler = StandardScaler()
    X_one_column = X_train.reshape([-1,1])
    result_one_column = scaler.fit_transform(X_one_column)
    X_train = (X_train - scaler.mean_)/(scaler.var_**0.5)
    

    X_train = X_train.reshape(X_train.shape[0],10,1)
    return scaler

def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _get_optimizer(_optimizer, learn_rate):
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

def train_model(conf):
    
    #print (psutil.Process().cpu_num())
    #proc_assigned = psutil.Process().cpu_num()
    #os.system("taskset -p -c 0-3 %d" % os.getpid())
    #exit()
    
    global X_train, y_train
    #def train_model(self,lstm_layer,epochs,batch_size,activation,learn_rate,dropout_rate,init,optimizer)
    """
    conf: [[lstm_layer], epochs, batch_size, activation, learn_rate, dropout_rate, init, optimizer]
                0           1           2           3           4           5         6       7
    """
    print("-------------",conf,"-----")
    #lstm_layer=0 
    """
    lstm_layer=0 
    dropout_rate=0.0
    activation = 'relu'
    optimizer='Adam'
    learn_rate=0.001
    init_mode='glorot_uniform'
    """

    s = '_'
    conf_file = s.join(conf)
    #print(type(conf_file))
    lstm_layer = list(map(int,conf[0].split(',')))
    epochs = int(conf[1])
    batch_size = int(conf[2])
    activation = conf[3]
    learn_rate = float(conf[4])
    dropout_rate = float(conf[5])
    init_mode = conf[6]
    optimizer = conf[7]
    #print("\n",list(map(int,test[0][0].split(','))))

    #print(lstm_layer, epochs, batch_size, activation, learn_rate, dropout_rate, init_mode, optimizer)      
    
    #exit()
    #print("shape_X",X_train.shape,"shape_Y",y_train.shape)
    model = Sequential()

	#-----------------------------------RETE NN------------------------------------------
    first_layer_size_LSTM= 16 if lstm_layer==0 else lstm_layer[0]
    second_layer_size_LSTM= 16 if lstm_layer==0 else lstm_layer[1]

    model.add(GRU(first_layer_size_LSTM, input_shape=(10,1), kernel_initializer=init_mode, activation=activation,return_sequences=False))
    model.add(Dropout(dropout_rate))
    #model.add(GRU(second_layer_size_LSTM, kernel_initializer=init_mode, activation=activation,return_sequences=False))
    #model.add(Dropout(dropout_rate))
    for layer_size in lstm_layer[1:]:
        model.add(Dense(layer_size, kernel_initializer=init_mode, activation= activation))
        model.add(Dropout(dropout_rate))
	#  model.add(Dense(neurons, input_dim=7, activation='relu'))
	#  model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	#--------------------------END---------RETE NN------------------------------------------
	
	
    _optimizer = _get_optimizer(optimizer, learn_rate)
    # Compile model
    #os.system("taskset -p -c "+str(proc_assigned)+" %d" % os.getpid())
    model.compile(loss='binary_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
    #print(model.summary())
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Final evaluation of the model
    #score = model.evaluate(X_test, y_test, verbose=0)
    
    #Y_pred = model.predict(X_test)
    #Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
    #confusion_matrix = metrics.confusion_matrix(y_test, Y_pred_round)
    scores = {
        'conf' : conf
    }
    
    model.save("../../RollingDB/Model/Grid_LSTM/"+conf_file+".h5")
    
    return scores

def test(test_path,grid_path,scenario,AllAttacks,scaler,name_value):
    os.system("taskset -p -c 0-3 %d" % os.getpid())
    #----------------------------------------------------TEST----------------------------------------------------------------
    
    for filename in glob.glob(os.path.join(grid_path, '*.h5')):
        print (os.path.basename(filename ))
        #print("------------------",score['conf'],"------------------")
        model = load_model(filename)
        for attack in AllAttacks:
            def _remove_negative(ds):
                return ds[ds > 0]

            DF_attack = pd.read_csv(test_path+attack)

            grouped_attack = DF_attack.groupby("Run")
            sim_lists = sorted(DF_attack.Run.unique())
            _simulations = len(sim_lists)
            attack_detect_delay = np.zeros( _simulations )
            attack_detect = 0
            fake_detect = 0   
            soon_detect = 0 
            #print(sim_lists)

            for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
                #print("----------------------------------------------------------------------------",simulation, end='\r')
                #print("--------",simulation, end='\r')
                data_attack = grouped_attack.get_group(simulation)

                grouped_values = data_attack.groupby("Value")
                name_values = sorted(data_attack.Value.unique())
                #print(name_values) 
                best_delay_values = 100
                name_value_detected = ' '
                flag_fake_detect = False

                                
                #print("---------------------------------------------",name_value)
                data_value = grouped_values.get_group(name_value[:-4])
                X_test = data_value.drop(['Run','Time','Start','Value','Detection'], axis=1).values
                Y_test = data_value['Detection'].values

                #X_test = scaler.transform(X_test)
                X_test = (X_test - scaler.mean_)/(scaler.var_**0.5)
                X_test = X_test.reshape((X_test.shape[0],10,1))
                Y_pred = model.predict(X_test)
                Y_pred_round = [1 * (x[0]>=0.95) for x in Y_pred]

                DF_single_value = data_value[['Time','Start','Detection']]
                DF_single_value = DF_single_value.assign(Pred = Y_pred_round) 
                
                early_detect = np.where(DF_single_value['Pred']>DF_single_value['Detection'])[0]
                early_detect = early_detect[early_detect>100]

                if len(early_detect)>0 and DF_single_value.iloc[early_detect[0]]['Time'] < DF_single_value.iloc[early_detect[0]]['Start']:
                    
                    if DF_single_value.iloc[early_detect[0]]['Time'] < 10:
                        soon_detect += 1
                    else:
                        flag_fake_detect = True
                    
                

                detection = np.where((DF_single_value['Pred'].astype(int)&DF_single_value['Detection'].astype(int))==1)[0]
                if len(early_detect)>0:
                    detection = early_detect #se sono qui - c'e stata una predizione di un attacco, successivo a Start

                #print(detection)
                if len(detection)>0:
                    delay_single_value = (DF_single_value.iloc[detection[0]]['Time']) - (DF_single_value.iloc[detection[0]]['Start'])

                    if delay_single_value < best_delay_values :
                        best_delay_values = delay_single_value
                        #name_value_detected = name_value
                    #print("delay_single_value", delay_single_value, " best_delay_values", best_delay_values)
                    
                if flag_fake_detect :
                    fake_detect += 1
                    attack_detect_delay[simulation_index] = -1
                elif best_delay_values != 100:
                    attack_detect += 1
                    attack_detect_delay[simulation_index] = best_delay_values
                    #print("WHO_detect",name_value_detected,best_delay_values)
                    
                else:
                    attack_detect_delay[simulation_index] = 0

                #print("fake_detect:", fake_detect, "attack_detect ",attack_detect, " attack_detect_delay", attack_detect_delay)
                #exit()
                
            print(attack,"->", " AD",attack_detect," FD",fake_detect, " SD", soon_detect, " delay", np.nan if len(_remove_negative(attack_detect_delay)) <= 0 else round(_remove_negative(attack_detect_delay).mean(),2))
    #--------------------------------------------------------------------------------------------------------------------
    

def main():
    train_path = "../../RollingDB/"
    test_path = "../../RollingDB/DB_Test/"
    grid_path = '../../RollingDB/Model/Grid_LSTM'
    scenario = "Random" #Constant
    #AllAttacks = ["{}NoInjection.csv".format(scenario),"{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    AllAttacks = ["{}NoInjection.csv".format(scenario)]
    name_value = "KFspeed.csv"
    
    files = glob.glob('../../RollingDB/Model/Grid_LSTM/*')
    for f in files:
        os.remove(f)
    
    
    start_time = time.time()

    scaler = get_XYscaler_Train(train_path, name_value)

    #########################(START) test one_by_one###########################################
    #dense_size_candidates = ['256,128,64', '32', '512,256,128,64', '256,128,128,64']
	
    lstm_size_candidates = ['64,32,32','128,128,256,128,64,32']
    epochs = [40,50] 
    batch_size = [32]    
    # Use scikit-learn to grid search 
    activation =  ['relu']
    learn_rate = [0.001]
    dropout_rate = [0.2,0.4]
    init = ['uniform']
    optimizer = ['Adam']
    #########################(END) test one_by_one#############################################
    
    configurations = np.array(np.meshgrid(lstm_size_candidates,epochs,batch_size,activation,learn_rate,dropout_rate,init,optimizer)).T.reshape(-1,8)
    print("CONFIGURATIONS: ",configurations,"\n",configurations.shape)
    time.sleep(2)

    num_workers = 16

    pool = multiprocessing.Pool(num_workers)
    scores = pool.map(train_model, configurations)

    print("---------END-----------------")
    
    test(test_path,grid_path,scenario,AllAttacks,scaler,name_value)

    print("--- %s s ---" % ((time.time() - start_time)))



if __name__ == "__main__":
  

    main()
       
    
    exit()  
