import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

col_train = ['KF distance', 'V2X-KF distance', 'V2X-KF speed', 'Radar distance', 'Radar-KF distance', 'Radar-V2X speed', 'Radar-KF speed', 'Detection']
col_test = ['KF distance', 'V2X-KF distance', 'V2X-KF speed', 'Radar distance', 'Radar-KF distance', 'Radar-V2X speed', 'Radar-KF speed','Run','Time','Start','Detection']
    

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

        self.vehicles = 8

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
            #_dt = 0
            _positions_comp = InjectionDetectionAnalyzer.__compensate_position(_dt, _positions, _speeds, _accelerations)
            _speeds_comp = InjectionDetectionAnalyzer.__compensate_speed(_dt, _speeds, _accelerations)
            #print("deltaT",_dt)
            """ _positions_comp = _positions + _dt * _speeds
            _speeds_comp = _speeds """

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

    def detection_analyzer(self, is_train):

        self._expected_distance = []
        self._pred_kf_kfEstimatedSpeedVar = []
        self._pred_data_V2XAcceleration = []
       

        _vehicle = 0
        
        _window = runningAvgWindow

        _sampling_times = self.sampling_times[_vehicle]

        _pred_data = self.vehicle_data[_vehicle]
        _pred_kf = self.kf_predictions[_vehicle]

        _foll_data = self.vehicle_data[_vehicle + 1]
        _foll_kf = self.kf_predictions[_vehicle + 1]
        #predData->positionX += predData->speed * (follData->time - predData->time);
        #_pred_data["V2XPositionComp"] += _pred_data["V2XSpeedComp"] *(_foll_data["RadarTime"] - _pred_data["RadarTime"])

        _v2x_distance = _pred_data["V2XPositionComp"] - _foll_data["V2XPosition"]
        _kf_distance = _pred_kf["kfEstimatedPositionComp"] - _foll_kf["kfEstimatedPosition"]
        self._expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(self.cacc_params, _foll_data["V2XSpeed"])

        _v2x_relative_speed = _pred_data["V2XSpeedComp"] - _foll_data["V2XSpeed"]
        _kf_relative_speed = _pred_kf["kfEstimatedSpeedComp"] - _foll_kf["kfEstimatedSpeed"]

     
        self._pred_kf_kfEstimatedSpeedVar = _pred_kf["kfEstimatedSpeedVar"]
        self._pred_data_V2XAcceleration = _pred_data["V2XAcceleration"]

        _data = []
        # KF Distance
        _data.append( {
            "values": _kf_distance - self._expected_distance - self.cacc_params["vehicle-length"],
        })
        #self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 1, **_data)
        #print("KF distance, a regime",_data["thresholds"][-1])
       
        # V2X Distance - KF Distance
        _data.append( {
            "values": InjectionDetectionAnalyzer.__running_avg(_v2x_distance - _kf_distance, _window),
        })
        #self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 2, **_data)
        #print("V2X-KF distance",_data["thresholds"][-1])
        

        # V2X Speed - KF Speed
        _data.append( {
            "values": InjectionDetectionAnalyzer.__running_avg(
                _pred_data["V2XSpeed"] - _pred_kf["kfEstimatedSpeed"], _window),
        })
        #_lines.append(self._plot_detection_line(ax, "C3", "V2X-KF speed" if legend else None, **_data))
        #print("V2X-KF speed",_data["thresholds"][-1])

        # Radar Distance
        _data.append( {
            "values": _foll_data["RadarDistance"] - self._expected_distance,
        })
        #_lines.append(self._plot_detection_line(ax, "C4", "Radar distance" if legend else None, **_data))
        #print("Radar distance",_data["thresholds"][-1])
        
        # Radar Distance - KF Distance
        _data.append( {
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance + self.cacc_params["vehicle-length"], _window),
        })
        #_lines.append(self._plot_detection_line(ax, "C5", "Radar-KF distance" if legend else None, **_data))
        #print("Radar-KF distance",_data["thresholds"][-1])

        # Radar Speed - V2X Speed
        _data.append( {
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _v2x_relative_speed, _window),
        })
        #_lines.append(self._plot_detection_line(ax, "C6", "Radar-V2X speed" if legend else None, **_data))
        #print("Radar-V2X speed",_data["thresholds"][-1])

        # Radar Speed - KF Speed
        _data.append( {
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _kf_relative_speed, _window),
        })
        #_lines.append(self._plot_detection_line(ax, "C7", "Radar-KF speed" if legend else None, **_data))
        #print("Radar-KF speed",_data["thresholds"][-1])

        stats_for_sim = self._get_stats_for_sim(_data, _sampling_times, col_train if is_train else col_test)
        if is_train is False:
            stats_for_sim.Time = (_sampling_times[window-1::window])[0:stats_for_sim.shape[0]]
            #print("STATS.TIME: ",stats_for_sim.Time)
            #print("Sample time: ",_sampling_times)
            stats_for_sim['Start'] = self.attack_start if self.attack_start is not None else 0
            stats_for_sim['Run'] = self.simulation_index
            #print("ROW: ",stats_for_sim.shape[0], "len firs col: ", len(stats_for_sim['KF distance']))
            #print("START: ",stats_for_sim)
            
        #global DF_train
        #DF_train = DF_train.append(stats_for_sim, ignore_index = True)
        return stats_for_sim

    def _get_stats_for_sim(self, _data, sampling_times, col ):
        global window
        #print("COLLLL",col)
        stats_for_sim = pd.DataFrame(columns=col)
        for _eq, _d in enumerate(_data):
            stats_for_sim[stats_for_sim.columns[_eq]] = self._set_eq_col(_eq, _d['values'], window)
            #break
        target_col = self._set_target_col(sampling_times,window)
        stats_for_sim[stats_for_sim.columns[-1]] = target_col
        #print("STATS FOR SIM\n", stats_for_sim)
        return stats_for_sim

    def _set_target_col(self,sampling_times,window):
        attack_start = self.attack_start
        target_array = np.zeros( int(len(sampling_times)/window) )
        
        if attack_start is None:
            return target_array

        print("type: ",type(sampling_times), " len: ",len(sampling_times), " attack_start: ", attack_start)
        #print("sample", sampling_times)
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

        print("direct",direct_index, "value direct:",sampling_times[direct_index])
        print("difference: ",sampling_times[direct_index]-attack_start)
        start = int(direct_index/window)
        
        target_array[start:] = 1
        print ("START: ", start, " END: ", len(target_array), " LAST: ", target_array[-1])
        return target_array

    def _set_eq_col(self, _eq, values, window ):   
        
        slopes = np.array(list(InjectionDetectionAnalyzer.__pairwise_map(InjectionDetectionAnalyzer.__get_slope, values, window)))

        return slopes

    @staticmethod
    def __get_slope(array):
        y = np.array(array)
        x = np.arange(len(y))        
        #slope, intercept, r_value, p_value, std_err = linregress(x,y)
        slope = ((x*y).mean() - x.mean()*y.mean()) / ((x**2).mean() - (x.mean())**2)
        return slope
    
    @staticmethod
    def __pairwise_map(func, items, window=10):
        iterators = [iter(items)] * window
        return map(func, zip(*iterators))


    @staticmethod
    def __compensate_position(dt, position, speed, acceleration):
        return position + dt * speed
        #return position + dt * speed + 0.5 * dt * dt * acceleration

    @staticmethod
    def __compensate_speed(dt, speed, acceleration):
        return speed
        #return speed + dt * acceleration

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
        _avg = np.zeros(len(array))
        _std = np.zeros(len(array))
        for _i in range(len(array)):
            _win = min(_i, window - 1)
            _avg[_i] = np.mean(array[_i - _win:_i + 1])
            _std[_i] = np.std(array[_i - _win:_i + 1])
        return (_avg, _std) if return_std else _avg

    @staticmethod
    def __compute_expected_distance(cacc_params, speed):
        return cacc_params["spacing"] + speed * cacc_params["headway"]

class CollectDataForAttack:
    def __init__(self, base_path, file_name):
        _path = os.path.join(base_path, file_name)
        #print(_path)
        self.all_data = pd.read_csv(_path, converters={
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
    base_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/summary/"#../InjectionDetectionData
    scenario = "Random" #Constant
    controller = "CACC" #Test

    # base_path = os.path.join(base_path, controller)
    DF_train = pd.DataFrame(columns=col_train)
    export_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/"
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

        print("----------------DF TO EXPORT-----------------\n",DF_test)
        DF_test.to_csv(export_path+'/Test/'+attack,index=False, header=True)

    
    print("----------------DF TO EXPORT-----------------\n",DF_train)
    
    #EXPORT ORIGINAL
    DF_train.to_csv (export_path+'DB.csv', index = False, header=True)
    print(DF_train,"\n","MAX: \n",DF_train.max(), "\n MIN: \n", DF_train.min())
    # NORMALIZATION
    mms = StandardScaler()
    DF_train[DF_train.columns[0:-1]] = mms.fit_transform(DF_train[DF_train.columns[0:-1]])
    #print(to_export_norm)
    # EXPORT NORMALIZED TO CSV
    DF_train.to_csv (export_path+'DB_norm.csv', index = False, header=True)

    print("--- %s s ---" % ((time.time() - start_time)))

    exit()  
