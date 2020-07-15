import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from heatmap import heatmap, corrplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/"
    scenario = "Random" #Constant

    #NoAttack
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    #AllAttacks = ["{}CoordinatedInjection.csv".format(scenario)]
    for _attack_index, attack in enumerate(AllAttacks):
        DF_test = pd.DataFrame(columns=col_test)

        data_object = CollectDataForAttack(base_path, attack)
        test_data = data_object.get_data()
        grouped = test_data.groupby("run")
                                               #Range [start:stop] -> [start,stop)
        sim_lists = sorted(test_data.run.unique())
        _simulations = len(sim_lists)

        NoInjection = "NoInjection" in attack

        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            #print("-----------------------------------------------------------------------------------------------------------",simulation)
            data = grouped.get_group(simulation)
            analyzer = InjectionDetectionAnalyzer(data, simulation_index, NoInjection)
            is_train = True if simulation_index < 3 else False
            stats = analyzer.detection_analyzer(is_train)

            if is_train :
                DF_train = DF_train.append(stats, ignore_index = True)
            else:
                DF_test = DF_test.append(stats, ignore_index = True)

 

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
