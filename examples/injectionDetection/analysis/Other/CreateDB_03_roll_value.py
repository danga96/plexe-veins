import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import concat
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

col_train = ['v9', 'v8', 'v7', 'v6', 'v5', 'v4', 'v3', 'v2', 'v1', 'v0', 'Detection']
col_test = ['v9', 'v8', 'v7', 'v6', 'v5', 'v4', 'v3', 'v2', 'v1', 'v0','Run','Time','Start','Value','Detection']
    
train_simulation = 100

window = 10
#TODO: se si modifica window, occorre modificare anche le colonne di "col_train" e "col_test"

class DataTuning:

    def __init__(self, DB_values_path, export_path):
        DB_values = pd.read_csv(DB_values_path, converters={
            'time': DataTuning.__parse_ndarray,
            'value': DataTuning.__parse_ndarray
        })
        #print(DB_values)
        grouped = DB_values.groupby("attack")
        self.DB_values_train = {}
        self.DB_values_test = {} 
        self.name_values = (DB_values.name_value.unique())

        for i, name_value in enumerate(self.name_values):
            self.DB_values_train[name_value] = pd.DataFrame(columns=col_train)
            #self.DB_values_test[name_value] = pd.DataFrame(columns=col_test)

       

        attack_lists = (DB_values.attack.unique())
        _attacks = len(attack_lists)        

        for attack_index, attack in enumerate(attack_lists[1:]):#per ogni attacco
            print("-----------------------------------------------------------------------------------------------------------",attack)
            self.DB_values_test = pd.DataFrame(columns=col_test)
            self.attack_data = grouped.get_group(attack)

            self.split_DB()

            #print("----------------DF TO EXPORT-----------------",attack)
            self.DB_values_test.to_csv(export_path+'DB_Test/'+ attack +'.csv',index=False, header=True)


        print("-------------FRAME VALUE----------------------")
        for i, name_value in enumerate(self.name_values):
            print("----------------DF TO EXPORT-----------------",name_value)
            self.DB_values_train[name_value].to_csv(export_path+name_value+'.csv',index=False, header=True)
            #print(self.DB_values_train[name_value],"\n\n\n")
       


    @staticmethod
    def __parse_ndarray(value):
        value_temp = value.replace('[','').replace(']','')
        return np.fromstring(value_temp, sep=' ') if value else None


    def split_DB(self):
        global train_simulation
        grouped = self.attack_data.groupby("run")
                                               #Range [start:stop] -> [start,stop)
        sim_lists = sorted(self.attack_data.run.unique())
        _simulations = len(sim_lists)

       

        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            print("-------------------------------------------------------------------------------Simulation: ",simulation, end='\r')
            self.simulation_index = simulation_index
            self.simulation_data = grouped.get_group(simulation)
            is_train = True if simulation_index < train_simulation else False#minore di N :vuol dire che i primi N sono di train
            self.split_and_tuning(is_train)


        return None

    def split_and_tuning(self, is_train):
        global window
        self.attack_start = self.simulation_data['start'].iloc[0]
        #print("SIMULATION\n",self.simulation_data)
        #sampling_times = np.fromstring(self.simulation_data['time'].iloc[0].str.get(0), sep=' ')

        #print(self.simulation_data['time'], len(self.simulation_data['time']), type(self.simulation_data['time'].iloc[0]))
        
        """
        sampling_times =  self.simulation_data['time'].apply(lambda x: x.replace('[','').replace(']',''))
        sampling_times = np.fromstring(sampling_times.iloc[0], sep=' ')
        """
        #sampling_times = self.simulation_data['time'].loc['V2XKFspeed']
        sampling_times = self.simulation_data.loc[self.simulation_data.name_value == 'V2XKFspeed','time'].values[0]
        #print(sampling_times)
        
        for i, name_value in enumerate(self.name_values):# per ogni valore
            self.flag_not_diverge = False
            DF_temp_train = pd.DataFrame(columns=col_train)
            DF_temp_test = pd.DataFrame(columns=col_test)
            #print("\n\n",self.simulation_data['value'].iloc[i])
            #print("LEN:",len((self.simulation_data['value'].iloc[i])),"\n\n")
            data = self.simulation_data.loc[self.simulation_data.name_value == name_value,'value'].values[0]
            #data = self.simulation_data['value'].iloc[i]
            data_re = data.reshape(len(data),1)
            #print("Name_value",name_value," DATA: ", data, " MAX:", np.abs(data)[100:].max(), "mean: ", np.abs(data[100]), "sub: ",np.abs(data)[100:].max() - np.abs(data[100]))
            if (np.abs(data)[100:].max() - np.abs(data[100])) < 2:
                self.flag_not_diverge = True
            #exit()
            #print(data_re.shape,len(data_re),type(data_re))
            
            data_supervised = self.series_to_supervised(data.reshape(len(data),1),n_in=window-1)
            #print(data_supervised)
            
            DF_temp_train = DF_temp_train.append(data_supervised, ignore_index = True)
            target_col = self._set_target_col(sampling_times,window)
            #print("Target COL: ", target_col)
            #print("LEN_target_col",len(target_col))            
            DF_temp_train[DF_temp_train.columns[-1]] = target_col
            if is_train is False:
                DF_temp_test = DF_temp_test.append(DF_temp_train, ignore_index = True)
                #print(len(sampling_times[window-1:]))
                DF_temp_test.Time = sampling_times[window-1:]
                DF_temp_test['Start'] = self.attack_start if self.attack_start is not None else 0
                DF_temp_test['Run'] = self.simulation_index
                DF_temp_test['Value'] = name_value
                self.DB_values_test = self.DB_values_test.append(DF_temp_test, ignore_index=True)
            else:
                self.DB_values_train[name_value] = self.DB_values_train[name_value].append(DF_temp_train, ignore_index=True)
            #print(DF_temp_train)
            
            """
            if name_value == 'RV2Xspeed':
                #print(data_re.shape,len(data_re),type(data_re),"\n",data_re)
                print(DF_temp_train)
                exit()
            """
            #self.DB_values_train[name_value] = self.DB_values_train[name_value].append(DF_temp_train, ignore_index=True)
        #exit()
       

    def series_to_supervised(self, data, n_in=9, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """

        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('v%d' % (i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('v%d' % (j)) for j in range(n_vars)]
            else:
                names += [('v%d' % (i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        
        return agg



    def _set_target_col(self,sampling_times,window):
        

        attack_start = self.attack_start
        target_array = np.zeros( len(sampling_times)-window+1 )

        if attack_start == 0 or self.flag_not_diverge:
            return target_array

        #print("type: ",type(sampling_times), " len: ",len(sampling_times), " attack_start: ", attack_start)
        #print("sample", sampling_times)
    
        #print("MODULO",attack_start/0.1)
        direct_index = int(attack_start/0.1-10)
        direct_index -= 1 if sampling_times[direct_index]-attack_start > 0.1 else 0
        """#INDICE ACCESSO DIRETTO MANUALE
        print("atk st",int(attack_start/0.1-10))
        for _i, _v in enumerate(sampling_times):
            if _v > attack_start:
                _index = _i
                break
        print("index",_index," value",sampling_times[_index])
        """

        #print("direct",direct_index, "value direct:",sampling_times[direct_index])
        #print("difference: ",sampling_times[direct_index]-attack_start)
        start = direct_index-window+1
        
        target_array[start:] = 1
        #print ("START: ", start, " END: ", len(target_array), " LAST: ", target_array[-1])

        return target_array


if __name__ == "__main__":
    DB_values_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/DB_values.csv"
    export_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/"
    scenario = "Random" #Constant
    controller = "CACC" #Test
    start_time = time.time()

    
    data_tuning = DataTuning(DB_values_path,export_path)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
