import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random

col = ['attack','run','name_value','time','value','start']
    
if __name__ == "__main__":
    def _remove_negative(ds):
        return ds[ds > 0]

    base_path = "../../summary/"
    scenario = "Random" #Constant
    controller = "CACC" #Test

    # base_path = os.path.join(base_path, controller)
    test_path = "./RollingDB/DB_Test/"
    #NoAttack
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    #AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}CoordinatedInjection.csv".format(scenario)]

    DF_detect = pd.read_csv(test_path+'time_detect.csv')
    
    columns_max = [('%s%s' % (AllAttacks[x][6:-4],"_MAX")) for x in range(len(AllAttacks))]
    columns_min = [('%s%s' % (AllAttacks[x][6:-4],"_MIN")) for x in range(len(AllAttacks))]

    _features = []
    df = DF_detect.drop(['Run','Start','NoInjection'], axis=1)
   

    fig, ax = plt.subplots()
    ax.set_title('Distance in detection Time')
    AllAttacks = ['PositionInjection','SpeedInjection','AccelerationInjection','AllInjection','CoordinatedInjection']
    for attack in AllAttacks:
        df[attack] = DF_detect[attack] - DF_detect['Start']
    df = df.apply(_remove_negative)
    print("\n\n",df)

    boxplot = df.boxplot(grid=False)
    plt.show()
    
    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
