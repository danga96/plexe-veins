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

class DataAnalysis:
    def __init__(self, model_path, test_path, attack_temp):
        self.model = {}
        self.scaler = {}
        DF_temp = pd.read_csv(test_path+attack_temp)
        self.name_values = (DF_temp.Value.unique())

        print(attack_temp, self.name_values)

        for i, name_value in enumerate(self.name_values):
            self.model[name_value] = load_model(model_path+'model_'+name_value+'.h5')
            self.scaler[name_value] = joblib.load(model_path+'scaler_'+name_value+'.bin')


    def get_Models(self):
        return self.models

    def apply_model(self, model):
        
        self.model = model
        #model = LogisticRegression(C=1)        

        self.model.fit(self.X_train,self.Y_train)

        return None



    def get_stats_attack(self, test_path, attack):
        def _remove_negative(ds):
            return ds[ds > 0]

        DF_attack = pd.read_csv(test_path+attack)

        """
        X_test = self.X_test[attack]
        Y_test = self.Y_test[attack]
        DF_test = self.DF_test_collection[attack]

        X_test = self.scaler.transform(X_test)

        Y_pred = self.model.predict(X_test)
        DF_drop_values = DF_test[['Run','Time','Start','Detection']]
        DF_drop_values = DF_drop_values.assign(Pred = Y_pred) 

        #DF_drop_values['Pred'] = Y_pred
        #print(DF_drop_values)
        
        #Y_pred_proba = self.model.predict_proba(X_test)
        
        #print("Y_pred: \n",Y_pred)
        #print("Y_pred_proba: \n",Y_pred_proba)

        #print("ACCURACY: ", str(metrics.accuracy_score(Y_test,Y_pred)))
        #print("LOG LOSS: ", str(metrics.log_loss(Y_test, Y_pred_proba, labels=[0.0,1.0])))

        #print(classification_report(Y_test, Y_pred))
        confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
        print(confusion_matrix)

        #print("Y_test", Y_test,"len y_test: ", len(Y_test), " \nY_pred: ", Y_pred, "len y_pred: ", len(Y_pred))
        """
        grouped_attack = DF_attack.groupby("Run")
        sim_lists = sorted(DF_attack.Run.unique())
        _simulations = len(sim_lists)
        attack_detect_delay = np.zeros( _simulations )
        attack_detect = 0
        fake_detect = 0         
        
        leader_attack_detect = 0
        predecessor_attack_detect = 0
        leader_attack_detect_delay = np.zeros( _simulations )
        predecessor_attack_detect_delay = np.zeros( _simulations )
        attack_detect = np.zeros( 7 )        
        
        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            #print("-----------------------------------------------------------------------------------------------------------",simulation)
            #print("--------",simulation, end="\r", flush=True)
            data_attack = grouped_attack.get_group(simulation)

            grouped_values = data_attack.groupby("Value")
            name_values = sorted(data_attack.Value.unique())
            for name_value in name_values:
                data_value = grouped_values.get_group(name_value)
                X_test = data_value.drop(['Run','Time','Start','Value','Detection'], axis=1).values
                Y_test = data_value['Detection'].values

                X_test = self.scaler[name_value].transform(X_test)
                X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
                Y_pred = self.model[name_value].predict(X_test)
                Y_pred_round = [1 * (x[0]>=0.9) for x in Y_pred]
                DF_drop_values = data_value[['Time','Start','Detection']]
                DF_drop_values = DF_drop_values.assign(Pred = Y_pred) 
                #print("\nX_TEST",X_test[0],"\n")
                #print("\nY_TEST",Y_test,"\n")
                #print("\nDF_drop_values\n",DF_drop_values,"\n")
                #print("\nY_pred",Y_pred,"\n")
                if name_value == 'Rdistance':
                    print("\nY_pred",DF_drop_values[0:30],"\n")
                print(name_value)
                confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred_round)
                print(confusion_matrix)

                

            exit()
            flag_fake_detect = False
            #print(data,"\n\n\n\n")
            early_detect = np.where(data['Pred']>data['Detection'])[0]
            if len(early_detect)>0:
                #print(data.iloc[early_detect[0]]['Start'])
                fake_detect += 1
                flag_fake_detect = True

            #TODO: andrebbe un IF per evitare di contare come attacchi rilevati anche i fake attacks
            detection = np.where((data['Pred'].astype(int)&data['Detection'].astype(int))==1)[0]
            #print(detection)
            if len(detection)>0:
                #print(data.iloc[detection[0]])
                attack_detect_delay[simulation_index] = (data.iloc[detection[0]]['Time']) - (data.iloc[detection[0]]['Start'])
                #print("Delay", attack_detect_delay)
                attack_detect += 1
            else:
                attack_detect_delay[simulation_index] = 0

            attack_detect_delay[simulation_index] = -1 if flag_fake_detect else attack_detect_delay[simulation_index]
            #exit()
        print("------------- ",attack," (TOT_SIM: ",_simulations,") -------------")
        #print("ad: ",attack_detect_delay)
        print("DETECTION")
        print("  attack_detect: ",attack_detect, " ({:.2f}%)".format(100 * attack_detect/_simulations), 
                " false_positive: ", fake_detect, " ({:.2f}%)".format(100 * fake_detect/_simulations),
                    " delay(s): ", np.nan if len(_remove_negative(attack_detect_delay)) <= 0 else 
                                            round(_remove_negative(attack_detect_delay).mean(),2))
        #exit()
                
        

if __name__ == "__main__":
    model_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/Model/"
    test_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/DB_Test/"
    scenario = "Random" #Constant
    w_radar = False
    #NoAttack
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    AllAttacks = ["{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    analyzer = DataAnalysis(model_path,test_path,AllAttacks[0])

    for attack in AllAttacks:
        print("--------------------------------------------",attack,"-------------------------------------")
        analyzer.get_stats_attack(test_path, attack)

        for _attack_index, attack in enumerate(AllAttacks):
            analyzer.get_stats_attack(attack)
        
        #exit() 

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
