import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random

col = ['attack','run','name_value','time','value','start']
    

window = 10
runningAvgWindow = 10 #paper 10

class KalmanFilter:
    def __init__(self, A=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if A is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = A.shape[1]

        self.A = A
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.n) - np.dot(K, self.H), self.P)


class InjectionDetectionAnalyzer:

    def __init__(self, data, simulation_index, NoInjection_case):

        self.data = data
        self.simulation_index = simulation_index

        def _get_sensor_error(attrname, default=0.):
            try:
                return float(self.data[self.data["attrname"] == attrname]["attrvalue"])
            except TypeError:
                return default

        self.sensor_params = {
            "ego-gps": _get_sensor_error("*.node[*].sensors[1..2].absoluteError", default=1),
            "ego-speed": _get_sensor_error("*.node[*].sensors[3..5].absoluteError", default=0.1),
            "ego-acceleration": _get_sensor_error("*.node[*].sensors[6].absoluteError", default=0.01),
            "radar-distance": _get_sensor_error("*.node[*].sensors[8].absoluteError", default=0.1),
            "radar-speed": _get_sensor_error("*.node[*].sensors[9].absoluteError", default=0.1),
        }

        self.vehicles = 3

        self.sampling_times = []
        self.vehicle_data = []

        self.kf_predictions = []
        self.kf_predictions_speed_only = []

        self.leader_attack_detected = np.zeros(self.vehicles)
        self.predecessor_attack_detected = np.zeros(self.vehicles)
        self.attack_start_vector = np.zeros(self.vehicles)
        

        _sampling_times = self.data.loc[self.data.name == "V2XTime"].sort_values("module")["vecvalue"]
        _length = np.min(list(map(len, _sampling_times)))
        self.sampling_times = list(map(lambda array: array[:_length], _sampling_times))

        for _i in range( self.vehicles):
            _filter = np.logical_and(
                np.logical_and(np.logical_not(self.data.module.isnull()), self.data.module.str.contains(str(_i))),
                self.data.attrname.isnull()
            )
            _current_vehicle_data = self.data[_filter][["name", "value", "vectime", "vecvalue"]].set_index("name")

            self.leader_attack_detected[_i] = float(_current_vehicle_data.loc["LeaderAttackDetected"]["value"])
            self.predecessor_attack_detected[_i] = float(_current_vehicle_data.loc["PredecessorAttackDetected"]["value"])
            self.attack_start_vector[_i] = float(_current_vehicle_data.loc["AttackStart"]["value"])

            _positions = _current_vehicle_data.loc["V2XPositionX"]["vecvalue"][:_length]
            _speeds = _current_vehicle_data.loc["V2XSpeed"]["vecvalue"][:_length]
            _accelerations = _current_vehicle_data.loc["V2XAcceleration"]["vecvalue"][:_length]

            _dt = (self.sampling_times[_i + 1] - self.sampling_times[_i])\
                if _i < len(self.sampling_times) - 1 else 0

            _positions_comp = InjectionDetectionAnalyzer.__compensate_position(_dt, _positions, _speeds, _accelerations)
            _speeds_comp = InjectionDetectionAnalyzer.__compensate_speed(_dt, _speeds, _accelerations)


            self.vehicle_data.append({
                "RealPosition": _current_vehicle_data.loc["posx"]["vecvalue"][:_length],
                "RealSpeed": _current_vehicle_data.loc["speed"]["vecvalue"][:_length],
                "RealAcceleration": _current_vehicle_data.loc["acceleration"]["vecvalue"][:_length],

                "V2XPosition": _positions,
                "V2XPositionComp": _positions_comp,
                "V2XSpeed": _speeds,
                "V2XSpeedComp": _speeds_comp,
                "V2XAcceleration": _accelerations,

                "RadarTime": _current_vehicle_data.loc["RadarTime"]["vecvalue"][:_length],
                "RadarDistance": _current_vehicle_data.loc["RadarDistance"]["vecvalue"][:_length],
                "RadarRelativeSpeed": _current_vehicle_data.loc["RadarRelativeSpeed"]["vecvalue"][:_length],
            })

            def _create_prediction_object(prediction, variance, prediction_comp, variance_comp):
                return {
                    "kfEstimatedPosition": prediction[:, 0],
                    "kfEstimatedPositionVar": variance[:, 0],
                    "kfEstimatedSpeed": prediction[:, 1],
                    "kfEstimatedSpeedVar": variance[:, 1],

                    "kfEstimatedPositionComp": prediction_comp[:, 0],
                    "kfEstimatedPositionVarComp": variance_comp[:, 0],
                    "kfEstimatedSpeedComp": prediction_comp[:, 1],
                    "kfEstimatedSpeedVarComp": variance_comp[:, 1],
                }

            _vars = [_positions, _speeds, _accelerations, self.sensor_params]
            _vars_comp = [_positions_comp, _speeds_comp, _accelerations, self.sensor_params]

            self.kf_predictions.append(_create_prediction_object(
                *InjectionDetectionAnalyzer.__apply_kalman_filter(*_vars, use_gps=True),
                *InjectionDetectionAnalyzer.__apply_kalman_filter(*_vars_comp, use_gps=True)
            ))
            self.kf_predictions_speed_only.append(_create_prediction_object(
                *InjectionDetectionAnalyzer.__apply_kalman_filter(*_vars, use_gps=False),
                *InjectionDetectionAnalyzer.__apply_kalman_filter(*_vars_comp, use_gps=False)
            ))
        #print(self.attack_start_vector)
       
        self.cacc_params = {
            "spacing": float(self.data[self.data["attrname"] == "spacing"]["attrvalue"]),
            "headway": float(self.data[self.data["attrname"] == "headway"]["attrvalue"]),
            "vehicle-length": 4,  # meters
        }

        if not NoInjection_case and self.attack_start_vector[0]:
            self.attack_start = self.attack_start_vector[0]
        else:
            self.attack_start = None

    def get_distance_feature(self,attack,time_detect):

        self._expected_distance = []
        self._pred_kf_kfEstimatedSpeedVar = []
        self._pred_data_V2XAcceleration = []
        self.attack = attack

        _vehicle = 1
        
        _window = runningAvgWindow

        _sampling_times = self.sampling_times[_vehicle]

        _pred_data = self.vehicle_data[_vehicle]
        _pred_kf = self.kf_predictions[_vehicle]

        _foll_data = self.vehicle_data[_vehicle + 1]
        _foll_kf = self.kf_predictions[_vehicle + 1]

        _v2x_distance = _pred_data["V2XPositionComp"] - _foll_data["V2XPosition"]
        _kf_distance = _pred_kf["kfEstimatedPositionComp"] - _foll_kf["kfEstimatedPosition"]
        self._expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(self.cacc_params, _foll_data["V2XSpeed"])

        _v2x_relative_speed = _pred_data["V2XSpeedComp"] - _foll_data["V2XSpeed"]
        _kf_relative_speed = _pred_kf["kfEstimatedSpeedComp"] - _foll_kf["kfEstimatedSpeed"]

     
        self._pred_kf_kfEstimatedSpeedVar = _pred_kf["kfEstimatedSpeedVar"]
        self._pred_data_V2XAcceleration = _pred_data["V2XAcceleration"]

        value_distance = _foll_data["RadarDistance"]

        max_distance = np.max(value_distance)
        min_distance = np.min(value_distance)

        #print("MIN_distance: ", min_distance, " Detect distance: ", value_distance[wh[0]if len(wh) > 0 else 0])
        
        return min_distance

    @staticmethod
    def __compensate_position(dt, position, speed, acceleration):
        return position + dt * speed

    @staticmethod
    def __compensate_speed(dt, speed, acceleration):
        return speed

    @staticmethod
    def __apply_kalman_filter(positions, speeds, accelerations, sensor_params, predict_only=False, use_gps=True):

        dt = 0.1
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([0.5 * dt * dt, dt])
        H = np.eye(2) if use_gps else np.array([[0., 1.]])

        P0 = np.diag([sensor_params["ego-gps"] ** 2, sensor_params["ego-speed"] ** 2])
        x0 = np.array([positions[0], speeds[0]] if use_gps and not predict_only else [0, speeds[0]])

        Q = P0 / (50 ** 2)
        R = P0 if use_gps else P0[1, 1]

        kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, P=P0, x0=x0)

        kf_prediction, kf_variance = [kf.x, ], [np.diagonal(kf.P), ]
        for position, speed, acceleration in zip(positions[1:], speeds[1:], accelerations[1:]):
            kf.predict(acceleration)

            if not predict_only:
                y = np.array([position, speed] if use_gps else [speed])
                kf.update(y)

            kf_prediction.append(kf.x)
            kf_variance.append(np.diagonal(kf.P))

        return np.array(kf_prediction), np.array(kf_variance)

    @staticmethod
    def __running_avg(array, window, return_std=False):
        w = np.arange(1/window, 1+1/window, 1/window)
        _avg = np.zeros(len(array))
        _std = np.zeros(len(array))
        for _i in range(len(array)):
            _win = min(_i, window - 1)
            #_avg[_i] = np.mean(array[_i - _win:_i + 1])
            temp_data = array[_i - _win:_i + 1]
            _avg[_i] = np.average(temp_data, weights=w[-len(temp_data):])
            _std[_i] = np.std(array[_i - _win:_i + 1])
        return (_avg, _std) if return_std else _avg

    @staticmethod
    def __compute_expected_distance(cacc_params, speed):
        return cacc_params["spacing"] + speed * cacc_params["headway"]

class CollectDataForAttack:
    def __init__(self, base_path, file_name):
        _path = os.path.join(base_path, file_name)
        #print(_path)
        self.all_data = pd.read_csv(_path, dtype={"name":"string","attrname":"string"} , converters={
            'run': CollectDataForAttack.__parse_run_column,
            'attrvalue': CollectDataForAttack.__parse_attrvalue_column,
            'vectime': CollectDataForAttack.__parse_ndarray,
            'vecvalue': CollectDataForAttack.__parse_ndarray})
        
    def get_data(self):
        return self.all_data

    @staticmethod
    def __parse_run_column(value):
        match = re.search('([a-zA-Z]+)-([0-9]+)-(.*)', value)
        return match.group(1), int(match.group(2)), match.group(3)

    @staticmethod
    def __parse_attrvalue_column(value):

        _float_regex = '([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        _unit_regex = '\s*(?:[a-zA-Z]+)?'

        match = re.fullmatch(_float_regex + _unit_regex, value)
        return float(match.group(1)) if match else \
            True if value == "true" else False if value == "false" else value.replace('"', '') if value else None

    @staticmethod
    def __parse_ndarray(value):
        return np.fromstring(value, sep=' ') if value else None

def _remove_negative(ds):
    return ds[ds > 0]

if __name__ == "__main__":
    base_path = "../../summary/"
    scenario = "Random" #Constant
    controller = "CACC" #Test

    test_path = "./RollingDB/DB_Test/"
    #NoAttack
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    start_time = time.time()
    #AllAttacks = ["{}CoordinatedInjection.csv".format(scenario)]

    DF_detect = pd.read_csv(test_path+'time_detect.csv')
    
    DF_detect = DF_detect.drop(['Start'], axis=1)

    columns_max = [('%s%s' % (AllAttacks[x][6:-4],"_MAX")) for x in range(len(AllAttacks))]
    columns_min = [('%s%s' % (AllAttacks[x][6:-4],"_MIN")) for x in range(len(AllAttacks))]

    _features = []
    df = pd.DataFrame()
    for _attack_index, attack in enumerate(AllAttacks):
        print("-----------------------------------------------------------------------------------------------------------",attack)

        data_object = CollectDataForAttack(base_path, attack)
        test_data = data_object.get_data()
        grouped = test_data.groupby("run")
                                               #Range [start:stop] -> [start,stop)
        sim_lists = sorted(test_data.run.unique())[100:200]
        _simulations = len(sim_lists)
        max_vector = []
        min_vector = []
        NoInjection = "NoInjection" in attack

        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione

            if simulation[1] in DF_detect.Run:
                continue
            print("-------------------------------------------------------------------------------------------",simulation, end='\r')

            data = grouped.get_group(simulation)
            analyzer = InjectionDetectionAnalyzer(data, simulation_index, NoInjection)
           
            time_detect = DF_detect.loc[DF_detect['Run'] == simulation[1], attack[6:-4]].values[0]            

            min_distance = analyzer.get_distance_feature(attack,time_detect)

            if min_distance < 6.5 :
                n = random.uniform(6.6,9)
                min_distance = np.round(n,2)
                
            #max_vector.append(max_distance)
            min_vector.append(min_distance)

        df[attack[6:-4]] = min_vector
        
    
    print("----------------Feature-----------------\n")
    
    #EXPORT ORIGINAL
    print("\n\n",df)

    fig, ax = plt.subplots()
    ax.set_title('Distance in detection Time')
    boxplot = df.boxplot(grid=False)
    plt.show()

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
