import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from heatmap import heatmap, corrplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model

DF_detect = {}

class DataAnalysis:
    def __init__(self, model_path, test_path, attack_temp):
        global DF_detect
        DF_detect = pd.DataFrame()
        self.w_radar = True
        self.model = {}
        self.scaler = {}
        DF_temp = pd.read_csv(test_path+attack_temp)
        self.name_values = (DF_temp.Value.unique())

        #print(attack_temp, self.name_values)

        for i, name_value in enumerate(self.name_values):
            self.model[name_value] = load_model(model_path+'model_'+name_value+'.h5')
            self.scaler[name_value] = joblib.load(model_path+'scaler_'+name_value+'.bin')


    def get_stats_attack(self, test_path, attack):
        global DF_detect
        def _remove_negative(ds):
            return ds[ds > 0]

        DF_attack = pd.read_csv(test_path+attack)

        grouped_attack = DF_attack.groupby("Run")
        sim_lists = sorted(DF_attack.Run.unique())

        DF_detect = DF_detect.assign(Run = sim_lists)

        _simulations = len(sim_lists)
        attack_detect_delay = np.zeros( _simulations )
        time_detect = np.zeros( _simulations )
        attack_detect = 0
        fake_detect = 0   
        soon_detect = 0 
        who_is_fp = {}
        who_is_tp = {}
        for name_value in self.name_values:               
            who_is_fp[name_value] = 0
            who_is_tp[name_value] = 0
        start = 0
        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            print("----------------------------------------------------------------------------",simulation, end='\r')
            #if simulation>100:
            #    continue
            #print("--------",simulation, end='\r')
            data_attack = grouped_attack.get_group(simulation)

            grouped_values = data_attack.groupby("Value")
            name_values = sorted(data_attack.Value.unique())
            #print(name_values) 
            best_delay_values = 100
            name_value_detected = ' '
            flag_fake_detect = False

            for name_value in name_values:
                #if name_value != 'KFdistance':
                #if name_value!= 'V2XKFdistance':#TEST
                #if name_value != 'V2XKFspeed':
                #if name_value != 'Rdistance':
                #if name_value != 'RKFdistance':
                #if name_value != 'RV2Xspeed':
                #if name_value != 'RKFspeed':
                #if name_value != 'KFspeed':
                #if name_value == 'RKFdistance': 
                if name_value == 'Rdistance': 
                #if name_value == 'KFdistance': 
                    continue
                if name_value[0]=='R' and self.w_radar is False: #remove radar Value
                    continue
                #print("---------------------------------------------",name_value)
                data_value = grouped_values.get_group(name_value)
                X_test = data_value.drop(['Run','Time','Start','Value','Detection'], axis=1).values
                Y_test = data_value['Detection'].values
                #print("nv: ", name_value, " \nXTEST: ",X_test[10])
                #X_test = self.scaler[name_value].transform(X_test)
                X_test = (X_test - self.scaler[name_value].mean_)/(self.scaler[name_value].var_**0.5)
                #print("XTEST2: ",X_test[10])
                #X_test = (X_test - self.scaler[name_value].mean_)/(self.scaler[name_value].var_**0.5)
                
                X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
                Y_pred = self.model[name_value].predict(X_test)
                Y_pred_round = [1 * (x[0]>=0.95) for x in Y_pred]
                #print("\nY_pred",Y_pred[0:11],"\n")
                #exit()
                """
                for i in range(len(Y_pred[:-2])):
                    if Y_pred[i]>=0.80 and Y_pred[i+1]>=Y_pred[i]:
                        Y_pred_round[i] = 1
                    else:
                        Y_pred_round[i] = 0
                """
                """
                if simulation_index == 13:
                    for i in range(len(Y_pred)):
                        print(" ",Y_pred[i],"\t",Y_pred_round[i])
                """
                DF_single_value = data_value[['Time','Start','Detection']]
                DF_single_value = DF_single_value.assign(Pred = Y_pred_round) 
                start = DF_single_value['Start'].iloc[0]
                ##print(Y_pred, "\n\n", X_test[0])
                ##exit()
                #print("\nX_TEST",X_test[0],"\n")
                #print("\nY_TEST",Y_test,"\n")
                #print("\nDF_single_value\n",DF_single_value,"\n")
                #print("\nY_pred",Y_pred,"\n")
                
                
                """
                if name_value == 'Rdistance':
                    #continue
                    print("\nY_pred\n",DF_single_value[0:30],"\n")
                #print(name_value)
                confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred_round)
                print(confusion_matrix)
                """

                early_detect = np.where(DF_single_value['Pred']>DF_single_value['Detection'])[0]
                #print("Early detect",early_detect, end='\t')
                #print("PRED:\n",DF_single_value['Pred'],"Det: \n",DF_single_value['Detection'])
                """
                for time in early_detect:
                    if time > 100:
                        early_detect[0] = time
                        break
                """
                early_detect = early_detect[early_detect>100] #rimuove le DETECTION prima dei 100 secondi
                
                #print("Early",early_detect)
                #exit()
                
                    
                if len(early_detect)>0 and DF_single_value.iloc[early_detect[0]]['Time'] < DF_single_value.iloc[early_detect[0]]['Start']:
                    
                    if DF_single_value.iloc[early_detect[0]]['Time'] < 10:
                        soon_detect += 1
                    else:
                    #else:
                    #    print(DF_single_value.iloc[early_detect[0]]['Time']," ",name_value," ",DF_single_value.iloc[early_detect[0]]['Start'])
                    #fake_detect += 1
                        who_is_fp[name_value] += 1
                        flag_fake_detect = True
                        break
                

                detection = np.where((DF_single_value['Pred'].astype(int)&DF_single_value['Detection'].astype(int))==1)[0]
                if len(early_detect)>0:
                    #print("Detection_first",detection)
                    detection = early_detect #se sono qui => c'è stata una predizione di un attacco, successivo a Start (0 1)
                    #print("Detection_second",detection)
                    #exit()
                #print(detection)
                if len(detection)>0:
                    delay_single_value = (DF_single_value.iloc[detection[0]]['Time']) - (DF_single_value.iloc[detection[0]]['Start'])
                    #if "NoInjection" in attack:
                    #    print(DF_single_value.iloc[detection[0]]['Time']," ",name_value," ",DF_single_value.iloc[detection[0]]['Start'],simulation_index)
                    #print(np.array(DF_single_value['Pred']))
                    #for i in range(len(Y_pred)):
                    #    print(" ",Y_pred[i],"\t",Y_pred_round[i])
                    #exit()
                    #time.sleep(10)
                    #exit()
                    if delay_single_value < best_delay_values :
                        best_delay_values = delay_single_value
                        name_value_detected = name_value
                        best_time_detect = DF_single_value.iloc[detection[0]]['Time']
                    #print("delay_single_value", delay_single_value, " best_delay_values", best_delay_values)
                
            if flag_fake_detect :
                fake_detect += 1
                attack_detect_delay[simulation_index] = -1
            elif best_delay_values != 100:
                attack_detect += 1
                attack_detect_delay[simulation_index] = best_delay_values
                time_detect[simulation_index] = best_time_detect
                #print("WHO_detect",name_value_detected,best_delay_values)
                who_is_tp[name_value_detected] += 1
            else:
                attack_detect_delay[simulation_index] = 0

            #print("fake_detect:", fake_detect, "attack_detect ",attack_detect, " attack_detect_delay", attack_detect_delay)
            #exit()
            
            DF_detect.loc[DF_detect['Run'] == simulation,'Start']  = start
            
        print("------------- ",attack," (TOT_SIM: ",_simulations,") -------------")
        #print("ad: ",attack_detect_delay)
        print("DETECTION")
        print("  attack_detect: ",attack_detect, " ({:.2f}%)".format(100 * attack_detect/_simulations), 
                " false_positive[soon_detect]: ", fake_detect," [",soon_detect,"] ", " ({:.2f}%)".format(100 * fake_detect/_simulations),
                    " delay(s): ", np.nan if len(_remove_negative(attack_detect_delay)) <= 0 else 
                                            round(_remove_negative(attack_detect_delay).mean(),2))

        print("WHO_FP",who_is_fp)
        print("WHO_TP",who_is_tp)
        #for key,value in who_is_fp:
           # print(who_is_fp)
        #exit()

        DF_detect[attack[6:-4]] = time_detect
                
        

if __name__ == "__main__":
    model_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/"
    test_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/DB_Test/"
    scenario = "Random" #Constant

    #NoAttack
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    #AllAttacks = ["{}NoInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    #AllAttacks = ["{}CoordinatedInjection.csv".format(scenario)]
    
    analyzer = DataAnalysis(model_path,test_path,AllAttacks[0])

    for attack in AllAttacks:
        print("--------------------------------------------",attack,"-------------------------------------")
        analyzer.get_stats_attack(test_path, attack)

    print(DF_detect)
    DF_detect.to_csv(test_path+'time_detect.csv',index=False, header=True)
    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
