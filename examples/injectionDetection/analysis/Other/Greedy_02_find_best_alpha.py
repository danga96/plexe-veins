import re
import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

summary_detect = []
best_alpha_sims = []
best_alpha_attacks = []

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

    def __init__(self, data, detection_parameters, simulation_index, NoInjection_case, use_prediction=False):

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

        self.vehicles = 2
        self.detection_parameters = detection_parameters

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
                #if _i < len(self.sampling_times) - 1 and use_prediction else 0
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

    def detection_analyzer(self):

        #if self.attack_start_vector[0]:
        #    ax.axvline(self.attack_start_vector[0], color="black", linestyle=":", lw=1)
        self._set_alpha_parameters()

        self._expected_distance = []
        self._kf_position_std = []
        self._kf_speed_std = []
        self._pred_kf_kfEstimatedSpeedVar = []
        self._pred_data_V2XAcceleration = []
       

        _vehicle = 0
        _window = self.detection_parameters["runningAvgWindow"]

        _sampling_times = self.sampling_times[_vehicle]

        _pred_data = self.vehicle_data[_vehicle]
        _pred_kf = self.kf_predictions[_vehicle]

        _foll_data = self.vehicle_data[_vehicle + 1]
        _foll_kf = self.kf_predictions[_vehicle + 1]
        #predData->positionX += predData->speed * (follData->time - predData->time);
        #_pred_data["V2XPositionComp"] += _pred_data["V2XSpeedComp"] *(_foll_data["RadarTime"] - _pred_data["RadarTime"])

        _v2x_distance = _pred_data["V2XPositionComp"] - _foll_data["V2XPosition"]
        _kf_distance = _pred_kf["kfEstimatedPositionComp"] - _foll_kf["kfEstimatedPosition"]
        self._kf_position_std = np.sqrt(_pred_kf["kfEstimatedPositionVarComp"]) + np.sqrt(_foll_kf["kfEstimatedPositionVar"])
        self._expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(self.cacc_params, _foll_data["V2XSpeed"])

        _v2x_relative_speed = _pred_data["V2XSpeedComp"] - _foll_data["V2XSpeed"]
        _kf_relative_speed = _pred_kf["kfEstimatedSpeedComp"] - _foll_kf["kfEstimatedSpeed"]
        self._kf_speed_std = np.sqrt(_pred_kf["kfEstimatedSpeedVarComp"]) + np.sqrt(_foll_kf["kfEstimatedSpeedVar"])

     
        self._pred_kf_kfEstimatedSpeedVar = _pred_kf["kfEstimatedSpeedVar"]
        self._pred_data_V2XAcceleration = _pred_data["V2XAcceleration"]

        _data = []
        # KF Distance
        _data.append( {
            "sampling_times": _sampling_times,
            "values": _kf_distance - self._expected_distance - self.cacc_params["vehicle-length"],
            "thresholds": self.detection_parameters["distanceKFThresholdFactor"] * self._expected_distance,
        })
        #self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 1, **_data)
        #print("KF distance, a regime",_data["thresholds"][-1])
       
        # V2X Distance - KF Distance
        _data.append( {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_v2x_distance - _kf_distance, _window),
            "thresholds": 3 * self._kf_position_std * self.detection_parameters["distanceV2XKFThresholdFactor"],
        })
        #self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 2, **_data)
        #print("V2X-KF distance",_data["thresholds"][-1])
        

        # V2X Speed - KF Speed
        _data.append( {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(
                _pred_data["V2XSpeed"] - _pred_kf["kfEstimatedSpeed"], _window),
            "thresholds": (self.sensor_params["ego-speed"] + np.sqrt(_pred_kf["kfEstimatedSpeedVar"]) * 3) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedV2XKFThresholdFactor"],
        })
        #_lines.append(self._plot_detection_line(ax, "C3", "V2X-KF speed" if legend else None, **_data))
        #print("V2X-KF speed",_data["thresholds"][-1])

        # Radar Distance
        _data.append( {
            "sampling_times": _sampling_times,
            "values": _foll_data["RadarDistance"] - self._expected_distance,
            "thresholds": self.detection_parameters["distanceRadarThresholdFactor"] * self._expected_distance,
        })
        #_lines.append(self._plot_detection_line(ax, "C4", "Radar distance" if legend else None, **_data))
        #print("Radar distance",_data["thresholds"][-1])
        
        # Radar Distance - KF Distance
        _data.append( {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance + self.cacc_params["vehicle-length"], _window),
            #"values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance, _window),
            "thresholds": (self.sensor_params["radar-distance"] + 3 * self._kf_position_std) *
                           self.detection_parameters["distanceRadarKFThresholdFactor"],
        })
        #_lines.append(self._plot_detection_line(ax, "C5", "Radar-KF distance" if legend else None, **_data))
        #print("Radar-KF distance",_data["thresholds"][-1])

        # Radar Speed - V2X Speed
        _data.append( {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _v2x_relative_speed, _window),
            "thresholds": (self.sensor_params["radar-speed"] + self.sensor_params["ego-speed"]*2) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedRadarV2XThresholdFactor"],
        })
        #_lines.append(self._plot_detection_line(ax, "C6", "Radar-V2X speed" if legend else None, **_data))
        #print("Radar-V2X speed",_data["thresholds"][-1])

        # Radar Speed - KF Speed
        _data.append( {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _kf_relative_speed, _window),
            "thresholds": (self.sensor_params["radar-speed"] + self._kf_speed_std * 3) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedRadarKFThresholdFactor"],
        })
        #_lines.append(self._plot_detection_line(ax, "C7", "Radar-KF speed" if legend else None, **_data))
        #print("Radar-KF speed",_data["thresholds"][-1])

        # Attack detected
        #ax.axvline(self.predecessor_attack_detected[1], color="red", linewidth=1, label="Detection" if legend else None)

        #summary_detect[self.simulation][7] = round(self.predecessor_attack_detected[1],4)
        self._optimal_thresholds(_data)

        summary_detect[self.simulation_index][7] = str(round(self.attack_start_vector[0],3))+"|"+str(round(self.predecessor_attack_detected[1],3))

        return None

    def _optimal_thresholds(self, _data ):
        for _eq, _d in enumerate(_data):
            __data = _d
            self._optimal_threshold(_eq+1, **__data)

        return None

    def _optimal_threshold(self, _eq, sampling_times, values, thresholds ): 
        attack_tolerance = self.detection_parameters["attackTolerance"]   
        attack_start = self.attack_start
        """
        Option 1: partire dal valore in attack start ed iterare da li in poi;
        Option 2: partire dall'inizio ed analizzare per tutti i sampling_time
        """

        """ #Option 1 (v1.2)
        _index = 0
        print("sample",len(sampling_times)," values",len(values))
        rd = attack_start%0.1
        print("rd",rd, " atk st",attack_start)
        for _i, _v in enumerate(sampling_times):
            if _v > attack_start:
                _index = _i
                break

        print("index",_index," value",sampling_times[_index])
         """
        if attack_start is not None :
            _best_detect = 100#TODO forse non serve
            _best_alpha = 1#TODO con orignale, invece di 1
            _attack_detected_original = InjectionDetectionAnalyzer.__compute_attack_start(sampling_times, values, thresholds, attack_tolerance)
            if _attack_detected_original is None:
                _attack_detected_original = 100
            for _i in range(0,len(sampling_times)):
                #print("threshold",thresholds[_i]," values",values[_i]," sampl",sampling_times[_i]," kf_pos_std",_kf_position_std[_i])
                #th_test = self._get_alpha_test(_eq,_i,values, thresholds)
                alpha_test = round(abs(values[_i]/(thresholds[_i]/self.alpha_parameters[_eq])),4)
                #print("new_th",alpha_test)
                #thresholds_test = self._set_th(_eq,alpha_test, thresholds)
                thresholds_test = alpha_test * (thresholds/self.alpha_parameters[_eq])
                #thresholds_beta = self._set_th_beta(_eq,alpha_test, thresholds)
                #if  np.array_equal(thresholds_test,thresholds_beta) is False :
                    #print(np.setdiff1d(thresholds_beta, thresholds_test))
                    #print("eq",_eq," _i",_i)
                    #exit()
                _attack_detected_test = InjectionDetectionAnalyzer.__compute_attack_start(sampling_times, values, thresholds_test, attack_tolerance)
                #print("det:",_attack_detected," nth:",alpha_test, " i:",_i)
                if _attack_detected_test is None:
                    _attack_detected_test = 0
                if _attack_detected_test>attack_start and _attack_detected_test<_attack_detected_original:
                    #se attack_detected_orignal è 100, vuol dire che sono riuscito a rilevare un attacco che con la vecchia soglia non veniva rilevato
                    if _attack_detected_test<_best_detect or alpha_test<_best_alpha:
                        _best_detect = _attack_detected_test
                        _best_alpha = alpha_test
                        #print("new_alpha_test",alpha_test)
                    #break
        else:
            _best_detect = 100
            _best_alpha = 1
            _attack_detected_original = 100
            #print("EQ:",_eq, "VAL:",np.amax(np.round(np.abs(values[10:]),2)))
            for _i in range(0,len(sampling_times)):
                alpha_test = round(abs(values[_i]/(thresholds[_i]/self.alpha_parameters[_eq])),4)
                thresholds_test = alpha_test * (thresholds/self.alpha_parameters[_eq])
                _attack_detected_test = InjectionDetectionAnalyzer.__compute_attack_start(sampling_times, values, thresholds_test, attack_tolerance)
                if _attack_detected_test is None and alpha_test<_best_alpha:
                    _best_alpha = alpha_test

        #print("best detect ",_best_detect, " best th:",_best_alpha)

        #print("simulation index",self.simulation_index, "   eq",_eq)
        best_alpha_sims[self.simulation_index+1][_eq-1] = round(_best_alpha,4)
        summary_detect[self.simulation_index][_eq-1] = str(round(_attack_detected_original,2))+"|"+str(round(_best_detect,2))
        return None

    """ 
    def _get_alpha_test(self, _eq, _i, values, threshold):
        switcher = {
            1: abs(values[_i]/(threshold[_i]/self.detection_parameters["distanceKFThresholdFactor"])),
            2: abs(values[_i]/(threshold[_i]/self.detection_parameters["distanceV2XKFThresholdFactor"])),
            3: abs(values[_i]/(threshold[_i]/self.detection_parameters["speedV2XKFThresholdFactor"])),
            4: abs(values[_i]/(threshold[_i]/self.detection_parameters["distanceRadarThresholdFactor"])),
            5: abs(values[_i]/(threshold[_i]/self.detection_parameters["distanceRadarKFThresholdFactor"])),
            6: abs(values[_i]/(threshold[_i]/self.detection_parameters["speedRadarV2XThresholdFactor"])),
            7: abs(values[_i]/(threshold[_i]/self.detection_parameters["speedRadarKFThresholdFactor"])),
        }
        return switcher.get(_eq, 0)
    
    def _set_th_beta(self, _eq, th_test, threshold):
        switcher = {
            1: th_test * (threshold/self.detection_parameters["distanceKFThresholdFactor"]),
            2: th_test * (threshold/self.detection_parameters["distanceV2XKFThresholdFactor"]),
        }
        return switcher.get(_eq, 0)
      
    def _set_th(self, _eq, th_test, threshold):
        th_test = round(th_test,2)
        arr1 = th_test*threshold
        arr2 = th_test*3 * self._kf_position_std
        if  _eq==2 and (np.array_equal(arr1,arr2) is False) :
            print(np.setdiff1d(arr1, arr2))
            print("stop",th_test)
            exit()
        switcher = {
            1: th_test * self._expected_distance,
            2: th_test * 3 * self._kf_position_std,
        }
        return switcher.get(_eq, 0)             
    """

    def _set_alpha_parameters(self):
        self.alpha_parameters = np.zeros(len(self.detection_parameters))

        self.alpha_parameters[1] = self.detection_parameters["distanceKFThresholdFactor"]
        self.alpha_parameters[2] = self.detection_parameters["distanceV2XKFThresholdFactor"]
        self.alpha_parameters[3] = self.detection_parameters["speedV2XKFThresholdFactor"]
        self.alpha_parameters[4] = self.detection_parameters["distanceRadarThresholdFactor"]
        self.alpha_parameters[5] = self.detection_parameters["distanceRadarKFThresholdFactor"]
        self.alpha_parameters[6] = self.detection_parameters["speedRadarV2XThresholdFactor"]
        self.alpha_parameters[7] = self.detection_parameters["speedRadarKFThresholdFactor"]

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
    def __compute_attack_start(sampling_times, values, thresholds, attack_tolerance):
        _attack_detected = np.abs(values) > thresholds

        _count = 0
        for _i, _v in enumerate(_attack_detected):
            _count = _count + 1 if _v else 0
            if _count == attack_tolerance:
                return sampling_times[_i]

        return None

    @staticmethod
    def __compute_expected_distance(cacc_params, speed):
        return cacc_params["spacing"] + speed * cacc_params["headway"]

class CollectDataForAttack:
    def __init__(self, base_path, file_name):
        _path = os.path.join(base_path, file_name)
        #print(_path)
        #self.all_data = pd.read_csv(_path , converters={
        self.all_data = pd.read_csv(_path, dtype={"name":"string","attrname":"string"} , converters={
            'run': CollectDataForAttack.__parse_run_column,
            'attrvalue': CollectDataForAttack.__parse_attrvalue_column,
            'vectime': CollectDataForAttack.__parse_ndarray,
            'vecvalue': CollectDataForAttack.__parse_ndarray})
        #print("INFO: \n\n",self.all_data.info(),"\n\n")


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

def diff_percentage_csv(columns):
    summary_alpha_percentage = []
    row = np.asarray(best_alpha_sims).shape[0]
    col = np.asarray(best_alpha_sims).shape[1]
    summary_alpha_percentage = np.zeros( (row, col) )
    for i in range(1,row):
        for j in range(col):
            summary_alpha_percentage[i][j] = ((best_alpha_sims[i-1][j]-best_alpha_sims[i][j])/best_alpha_sims[i-1][j])*100*(-1)

    test_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/test_percentage.csv"
    np.savetxt(test_path, summary_alpha_percentage, header=','.join(columns), fmt=",".join(["%f"] * (np.asarray(summary_alpha_percentage).shape[1])))
    return None

if __name__ == "__main__":

    base_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/summary/"#../InjectionDetectionData
    scenario = "Random" #Constant
    controller = "CACC" #Test

    detection_parameters = {
        "runningAvgWindow": 10,#paper 10
        "attackTolerance": 10,#paper 10

        "distanceKFThresholdFactor": 0.33,#paper 0.33
        "distanceRadarThresholdFactor": 0.25,#0.20 - paper 0.25
        "distanceV2XKFThresholdFactor": 1,#paper 1
        "distanceRadarKFThresholdFactor": 1,#1.5 - paper 1

        "speedV2XKFThresholdFactor": 1,#paper 1
        "speedRadarV2XThresholdFactor": 1,#paper 1
        "speedRadarKFThresholdFactor": 1,#paper 1

        "accelerationFactor": 0.05#paper 0.05
    }
    #print(detection_parameters)
    """ for key, value in detection_parameters.items():
        print("key ",key," value", value) """
    # base_path = os.path.join(base_path, controller)
    
    #NoAttack
    start_time = time.time()
    AllAttacks = ["{}NoInjection.csv".format(scenario),  "{}PositionInjection.csv".format(scenario), "{}SpeedInjection.csv".format(scenario),
                   "{}AccelerationInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario), "{}CoordinatedInjection.csv".format(scenario)]
    #AllAttacks = ["{}NoInjection.csv".format(scenario)]
    best_alpha_attacks = np.zeros( (len(AllAttacks), 7) )
    for _attack_index, attack in enumerate(AllAttacks):
        print("---------------------------------------------------------------------------------------------------",attack)
        data_object = CollectDataForAttack(base_path, attack)
        test_data = data_object.get_data()
        grouped = test_data.groupby("run")
                                                #Range [start:stop] -> [start,stop)
        sim_lists = sorted(test_data.run.unique())[100:]
        _simulations = len(sim_lists)

        summary_detect = [ [ " " for c in range( 8 ) ] 
                                    for r in range( _simulations ) ] 
        best_alpha_sims = np.zeros( (_simulations+1, 7) )

        best_alpha_sims[0] = [detection_parameters["distanceKFThresholdFactor"], 
                                detection_parameters["distanceV2XKFThresholdFactor"],
                                detection_parameters["speedV2XKFThresholdFactor"],
                                detection_parameters["distanceRadarThresholdFactor"],
                                detection_parameters["distanceRadarKFThresholdFactor"],
                                detection_parameters["speedRadarV2XThresholdFactor"],
                                detection_parameters["speedRadarKFThresholdFactor"]]
       
        
        row_sim = []

        NoInjection = "NoInjection" in attack

        for simulation_index, simulation in enumerate(sim_lists):#per ogni simulazione
            print("---------------------------------------------------------------------------------------------------",simulation, end='\r')
            data = grouped.get_group(simulation)
            analyzer = InjectionDetectionAnalyzer(data, detection_parameters, simulation_index, NoInjection)
            analyzer.detection_analyzer()
            row_sim.append("Sim{}".format(simulation_index))
            #simulation_index += 1
            
        columns = ['KF distance', 'V2X-KF distance', 'V2X-KF speed', 'Radar distance', 'Radar-KF distance', 'Radar-V2X speed', 'Radar-KF speed']
        rows = ["Original"]
        rows += row_sim
        print("type",type(best_alpha_sims))
        best_alpha_sims_DF = pd.DataFrame(data=best_alpha_sims, index=rows, columns=columns)
        best_alpha_sims_DF = best_alpha_sims_DF.replace(1.0, np.nan)
        a = np.array( [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] )
        best_alpha_sims_DF.loc["Original"] = a

        
        print("best_alpha_sims_DF \n",best_alpha_sims_DF)
        max_value =  best_alpha_sims_DF[columns].max(axis=0)
        #max_value =  best_alpha_sims_DF.loc(1:2,[columns]).max(axis=0)
        print("Max value:\n", max_value)
        best_alpha_attacks[_attack_index] = max_value
        print("HERE")
        f1, ax1 = plt.subplots(2,1)
        f1.canvas.set_window_title(attack)
        f1.suptitle(attack)
        f1.set_figwidth(12)
        
        #ax1.axis('tight')
        ax1[0].axis('off')
        #ax1.autoscale(enable=True, axis="Both")
              
        tab = ax1[0].table(cellText=best_alpha_sims, rowLabels=rows, colLabels=columns, loc="center")
        tab.auto_set_font_size(False)
        tab.auto_set_column_width([0,1,2,3,4,5,6,7])
        #tab.scale(1,1.5)
        tab.set_fontsize(11)
        """ EXPORT TO CSV
        diff_percentage_csv(columns)
        test_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/test.csv"
        np.savetxt(test_path, best_alpha_sims, header=','.join(columns), fmt=",".join(["%f"] * (np.asarray(best_alpha_sims).shape[1])))
        """
        ax1[1].axis('off')
        columns += ["Start|Pdetect"]
        rows.remove("Original")
        #ax1[1].autoscale(enable=True, axis="Both")         
        tab2 = ax1[1].table(cellText=summary_detect, rowLabels=rows, colLabels=columns, loc="center")
        tab2.auto_set_font_size(False)
        tab2.auto_set_column_width([0,1,2,3,4,5,6,7])
        tab2.scale(1,1.5)
        tab2.set_fontsize(11)

        f1.tight_layout()#adapt table in width

        
    print("BEST ALPHA ATTACKS\n", best_alpha_attacks)
    columns.remove("Start|Pdetect")
    best_alpha_attacks_DF = pd.DataFrame(data=best_alpha_attacks, index=AllAttacks, columns=columns)
    maxs =  best_alpha_attacks_DF[columns].max(axis=0)
    print("MAXXS\n",maxs)
    #plt.show()
    print("--- %s s ---" % ((time.time() - start_time)))
    exit()   