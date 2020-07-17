import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
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

class DataAnalysis:
    def __init__(self, train_path, test_path, AllAttacks, w_radar):
        self.DF_test_collection = {}
        self.X_test = {}
        self.Y_test = {}
        self.DF_train = pd.read_csv(train_path)
        if w_radar:
            self.X_train = self.DF_train.drop(["Detection"], axis=1).values
        else:
            self.X_train = self.DF_train.drop(self.DF_train.columns[[3,4,5,6,-1]], axis=1).values
        self.Y_train = self.DF_train["Detection"].values

        for attack in AllAttacks:
            self.DF_test_collection[attack] = pd.read_csv(test_path+attack)
            if w_radar:
                self.X_test[attack] = self.DF_test_collection[attack].drop(['Run','Time','Start','Detection'], axis=1).values
            else:
                self.X_test[attack] = self.DF_test_collection[attack].drop(self.DF_test_collection[attack].columns[[3,4,5,6,7,8,9,-1]], axis=1).values
            self.Y_test[attack] = self.DF_test_collection[attack]['Detection'].values

        self.models = []
        #self.models.append(('LR', LogisticRegression()))

        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        #self.models.append(('KNN', KNeighborsClassifier()))
        """
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC()))
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def get_Models(self):
        return self.models

    def apply_model(self, model):
        
        self.model = model
        #model = LogisticRegression(C=1)        

        self.model.fit(self.X_train,self.Y_train)

        return None



    def get_stats_attack(self, attack):
        def _remove_negative(ds):
            return ds[ds > 0]

        X_test = self.X_test[attack]
        Y_test = self.Y_test[attack]
        DF_test = self.DF_test_collection[attack]

        X_test = self.scaler.transform(X_test)

        Y_pred = self.model.predict(X_test)
        DF_drop_values = DF_test[['Run','Time','Start','Detection']]
        DF_drop_values = DF_drop_values.assign(Pred = Y_pred) 

        #DF_drop_values['Pred'] = Y_pred
        #print(DF_drop_values)
        Y_pred_proba = self.model.predict_proba(X_test)
        #print("Y_pred: \n",Y_pred)
        #print("Y_pred_proba: \n",Y_pred_proba)

        #print("ACCURACY: ", str(metrics.accuracy_score(Y_test,Y_pred)))
        #print("LOG LOSS: ", str(metrics.log_loss(Y_test, Y_pred_proba, labels=[0.0,1.0])))

        #print(classification_report(Y_test, Y_pred))
        confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
        print(confusion_matrix)

        #print("Y_test", Y_test,"len y_test: ", len(Y_test), " \nY_pred: ", Y_pred, "len y_pred: ", len(Y_pred))

        grouped = DF_drop_values.groupby("Run")
        sim_lists = sorted(DF_drop_values.Run.unique())
        _simulations = len(sim_lists)
        attack_detect_delay = np.zeros( _simulations )
        attack_detect = 0
        fake_detect = 0
        """ 
        
        leader_attack_detect = 0
        predecessor_attack_detect = 0
        leader_attack_detect_delay = np.zeros( _simulations )
        predecessor_attack_detect_delay = np.zeros( _simulations )
        attack_detect = np.zeros( 7 )        
        """
        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            #print("-----------------------------------------------------------------------------------------------------------",simulation)
            data = grouped.get_group(simulation)
            #print(data,"\n\n\n\n")
            early_detect = np.where(data['Pred']>data['Detection'])[0]
            if len(early_detect)>0:
                #print(data.iloc[early_detect[0]]['Start'])
                fake_detect += 1

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
            #exit()
        print("------------- ",attack," (TOT_SIM: ",_simulations,") -------------")
        print("ad: ",attack_detect_delay)
        print("DETECTION")
        print("  attack_detect: ",attack_detect, " ({:.2f}%)".format(100 * attack_detect/_simulations), 
                " false_positive: ", fake_detect, " ({:.2f}%)".format(100 * fake_detect/_simulations),
                    " delay(s): ", np.nan if len(_remove_negative(attack_detect_delay)) <= 0 else 
                                            round(_remove_negative(attack_detect_delay).mean(),2))
        #exit()
                
        

if __name__ == "__main__":
    train_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/DB.csv"
    test_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Test/"
    scenario = "Random" #Constant
    w_radar = False
    #NoAttack
    #AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
    #               "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    AllAttacks = ["{}AccelerationInjection.csv".format(scenario),"{}CoordinatedInjection.csv".format(scenario)]
    analyzer = DataAnalysis(train_path,test_path,AllAttacks, w_radar)

    for name, model in analyzer.get_Models():
        print("-----------------------",name,"-------------------------")
        analyzer.apply_model(model)

        for _attack_index, attack in enumerate(AllAttacks):
            analyzer.get_stats_attack(attack)
        
        #exit() 

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
