import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

summary_detect = np.zeros( (6, 8) )

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

    def __init__(self, base_path, file_name, detection_parameters, use_prediction=False, simulation=0):

        _path = os.path.join(base_path, file_name)
        #print(_path)
        _data = pd.read_csv(_path, converters={
            'run': InjectionDetectionAnalyzer.__parse_run_column,
            'attrvalue': InjectionDetectionAnalyzer.__parse_attrvalue_column,
            'vectime': InjectionDetectionAnalyzer.__parse_ndarray,
            'vecvalue': InjectionDetectionAnalyzer.__parse_ndarray})

        grouped = _data.groupby("run")
        print("enumerate",sorted(_data.run.unique())[simulation])
        self.data = grouped.get_group(sorted(_data.run.unique())[simulation])
        

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
        self.detection_parameters = detection_parameters

        self.sampling_times = []
        self.vehicle_data = []

        self.kf_predictions = []
        self.kf_predictions_speed_only = []

        self.leader_attack_detected = np.zeros(self.vehicles)
        self.predecessor_attack_detected = np.zeros(self.vehicles)
        self.attack_start_vector = np.zeros(self.vehicles)
        
        self.attack = 0

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

    def plot_detection_graph(self, title, ax=None, attack=0, legend=True):

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if title is not None:
            ax.set_title(title)
 
        print("-------------------------------------",title,"------------------")

        

        if self.attack_start_vector[0]:
            ax.axvline(self.attack_start_vector[0], color="black", linestyle=":", lw=1)

        self.attack = attack
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

        _lines = []
     
        self._pred_kf_kfEstimatedSpeedVar = _pred_kf["kfEstimatedSpeedVar"]
        self._pred_data_V2XAcceleration = _pred_data["V2XAcceleration"]
  
        # KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": _kf_distance - self._expected_distance - self.cacc_params["vehicle-length"],
            #"values":InjectionDetectionAnalyzer.__running_avg( _kf_distance - self._expected_distance - self.cacc_params["vehicle-length"], _window),
            "thresholds": self.detection_parameters["distanceKFThresholdFactor"] * self._expected_distance,
        }
        _lines.append(self._plot_detection_line(ax, "C1", "KF distance" if legend else None, **_data))
        print("KF distance",_data["thresholds"][-1])
       
        # V2X Distance - KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_v2x_distance - _kf_distance, _window),
            "thresholds": 3 * self._kf_position_std * self.detection_parameters["distanceV2XKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C2", "V2X-KF distance" if legend else None, **_data))
        print("_v2x_distance:",_v2x_distance - _kf_distance)
        print("NEW_AVG:",InjectionDetectionAnalyzer.__running_avg(_v2x_distance - _kf_distance, _window))
        print("V2X-KF distance",_data["thresholds"][-1])
        """         
        for i in range(len(_data["sampling_times"])):
            print("Time:",_data["sampling_times"][i]," pred: ",_pred_kf["kfEstimatedPositionComp"][i], " foll: ",_foll_kf["kfEstimatedPosition"][i])
        print("\n\n") 
        """
         
        # V2X Speed - KF Speed
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(
                _pred_data["V2XSpeed"] - _pred_kf["kfEstimatedSpeed"], _window),
            "thresholds": (self.sensor_params["ego-speed"] + np.sqrt(_pred_kf["kfEstimatedSpeedVar"]) * 3) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedV2XKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C3", "V2X-KF speed" if legend else None, **_data))
        print("V2X-KF speed",_data["thresholds"][-1])

        # Radar Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": _foll_data["RadarDistance"] - self._expected_distance,
            #"values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - self._expected_distance,_window),
            "thresholds": self.detection_parameters["distanceRadarThresholdFactor"] * self._expected_distance,
        }
        _lines.append(self._plot_detection_line(ax, "C4", "Radar distance" if legend else None, **_data))
        print("Radar distance",_data["thresholds"][-1])
        
        # Radar Distance - KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance + self.cacc_params["vehicle-length"], _window),
            #"values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance, _window),
            "thresholds": (self.sensor_params["radar-distance"] + 3 * self._kf_position_std) *
                           self.detection_parameters["distanceRadarKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C5", "Radar-KF distance" if legend else None, **_data))
        print("Radar-KF distance",_data["thresholds"][-1])

        # Radar Speed - V2X Speed
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _v2x_relative_speed, _window),
            "thresholds": (self.sensor_params["radar-speed"] + self.sensor_params["ego-speed"]*2) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedRadarV2XThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C6", "Radar-V2X speed" if legend else None, **_data))
        print("Radar-V2X speed",_data["thresholds"][-1])

        # Radar Speed - KF Speed
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarRelativeSpeed"] - _kf_relative_speed, _window),
            "thresholds": (self.sensor_params["radar-speed"] + self._kf_speed_std * 3) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedRadarKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C7", "Radar-KF speed" if legend else None, **_data))
        print("Radar-KF speed",_data["thresholds"][-1]) 
        #---------------------------ACCeleration
        #_lines.append(self._plot_acceleration_line(ax, "C8", "Acceleration" if legend else None, _sampling_times,InjectionDetectionAnalyzer.__running_avg(_pred_data["V2XAcceleration"],_window)))
        _lines.append(self._plot_other_line(ax, "C8", "Acceleration" if legend else None, _sampling_times,_pred_data["V2XAcceleration"]))
        #_lines.append(self._plot_other_line(ax, "C9", "V2XDistance" if legend else None, _sampling_times,InjectionDetectionAnalyzer.__running_avg(_kf_distance, _window)))
        _lines.append(self._plot_other_line(ax, "C10", "KFSpeed" if legend else None, _sampling_times,InjectionDetectionAnalyzer.__running_avg(_kf_relative_speed, _window)))
       
        
        # Attack detected
        ax.axvline(self.predecessor_attack_detected[1], color="red", linewidth=1, label="Detection" if legend else None)

        summary_detect[self.attack][7] = round(self.predecessor_attack_detected[1],4)

        summary_detect[0][0] = round(self.attack_start_vector[0],4)

        return np.array(_lines)

    def _plot_other_line(self, ax, color, label, sampling_times, values):
        value_smoothed = gaussian_filter1d(values, sigma=1)
        return [
            ax.plot(sampling_times, value_smoothed, color=color, label=label)[0],None,None,None
        ]

    def _plot_detection_line(self, ax, color, label, sampling_times, values, thresholds):
        _attack_detected = InjectionDetectionAnalyzer.__compute_attack_start(
            sampling_times, values, thresholds, self.detection_parameters["attackTolerance"])

        index = int(''.join(filter(str.isdigit, color)))
        summary_detect[self.attack][index-1] = _attack_detected
        value_smoothed = gaussian_filter1d(values, sigma=1)
        return [
            ax.plot(sampling_times, value_smoothed, color=color, label=label)[0] ,
            ax.plot(sampling_times, thresholds, color=color , alpha=0.75, linestyle=":")[0],
            ax.plot(sampling_times, -thresholds, color=color , alpha=0.75, linestyle=":")[0],
            ax.axvline(_attack_detected, color=color , linestyle=":") if _attack_detected is not None else None
        ]

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
        w = np.arange(0.1,1.1,0.1)
        _avg = np.zeros(len(array))
        _std = np.zeros(len(array))
        for _i in range(len(array)):
            _win = min(_i, window - 1)
            
            #_avg[_i] = np.mean(array[_i - _win:_i + 1])    #OLD_avg
            #-----NEW_avg
            temp_data = array[_i - _win:_i + 1]
            
            _avg[_i] = np.average(temp_data, weights=w[-len(temp_data):])
            #_avg[_i] = np.mean(array[_i - _win:_i + 1])
            """
            print(temp_data,"len",len(temp_data))
            print("AVG:",_avg[_i])
            if len(temp_data)==10:
                exit()
            """
            #----End---NEW_avg
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

    @staticmethod
    def setup_hide_lines(figure, lines, legend):

        _lined = dict()
        for _idx, _legend_line in enumerate(legend.get_lines()):
            _legend_line.set_picker(5)  # 5 pts tolerance
            _lined[_legend_line] = _idx
            
        def _on_line_pick(event):
            _fn_legend_line = event.artist
            _fn_line_idx = _lined[_fn_legend_line]

            _visible = False
            for _fn_line in lines[:, _fn_line_idx].flatten():
                if _fn_line is not None:
                    _fn_line.set_visible(not _fn_line.get_visible())
                    _visible = _fn_line.get_visible()

            # Change the alpha on the line in the legend so we can see what lines have been toggled
            _fn_legend_line.set_alpha(1.0 if _visible else 0.5)
            plt.draw()

        figure.canvas.mpl_connect('pick_event', _on_line_pick)


if __name__ == "__main__":

    base_path = "../../../../summary/"
    scenario = "Random" #Constant
    controller = "CACC" #Test

    detection_parameters = {
        "runningAvgWindow": 10,#paper 10
        "attackTolerance": 10,#paper 10
        
        "distanceKFThresholdFactor": 0.13,#paper 0.33
        "distanceV2XKFThresholdFactor": 0.83,#paper 1
        "speedV2XKFThresholdFactor": 0.84,#paper 1
        
        "distanceRadarThresholdFactor": 0.11,#0.20 - paper 0.25
        "distanceRadarKFThresholdFactor": 0.84,#1.5 - paper 1  
        "speedRadarV2XThresholdFactor": 0.72,#paper 1
        "speedRadarKFThresholdFactor": 0.97,#paper 1

        "accelerationFactor": 0.05#paper 0.05
    }
    #print(detection_parameters)
    """ #ORGINAL
        "distanceKFThresholdFactor": 0.33,#paper 0.33
        "distanceRadarThresholdFactor": 0.25,#0.20 - paper 0.25
        "distanceV2XKFThresholdFactor": 1,#paper 1
        "distanceRadarKFThresholdFactor": 1,#1.5 - paper 1

        "speedV2XKFThresholdFactor": 1,#paper 1
        "speedRadarV2XThresholdFactor": 1,#paper 1
        "speedRadarKFThresholdFactor": 1,#paper 1
    """
    """ for key, value in detection_parameters.items():
        print("key ",key," value", value) """
   # base_path = os.path.join(base_path, controller)
    #NoAttack
    _simulation = 0
    analyzers = {
        #"NoInjection": InjectionDetectionAnalyzer(base_path, "{}NoInjection.csv".format(scenario), detection_parameters, simulation=_simulation),
        #"PositionInjection": InjectionDetectionAnalyzer(base_path, "{}PositionInjection.csv".format(scenario), detection_parameters, simulation=_simulation),
        #"SpeedInjection": InjectionDetectionAnalyzer(base_path, "{}SpeedInjection.csv".format(scenario), detection_parameters, simulation=_simulation),
        #"AccelerationInjection": InjectionDetectionAnalyzer(base_path, "{}AccelerationInjection.csv".format(scenario), detection_parameters, simulation=_simulation),
        #"AllInjection": InjectionDetectionAnalyzer(base_path, "{}AllInjection.csv".format(scenario), detection_parameters, simulation=_simulation),
        "CoordinatedInjection": InjectionDetectionAnalyzer(base_path, "{}CoordinatedInjection.csv".format(scenario), detection_parameters, simulation=_simulation)
    }

    f1, ax1 = plt.subplots(len(analyzers), 1, sharex="all", num="{} Scenario - {} - Attack detection".format(scenario, controller))
    f1.suptitle("Attack detection")
    print(len(analyzers))
    f1_lines = []
    for i, key in enumerate(analyzers):                          #[i]    i
        f1_lines.append(analyzers[key].plot_detection_graph(key, ax1, attack=1 , legend=i == 0))
    
    InjectionDetectionAnalyzer.setup_hide_lines(f1, np.array(f1_lines), f1.legend())
    
    print( " summary ", summary_detect)
    f2, ax2 = plt.subplots(1,1, figsize=(14,2))
    f2.suptitle("Summary")
    columns = ['KF distance', 'V2X-KF distance', 'V2X-KF speed', 'Radar distance', 'Radar-KF distance', 'Radar-V2X speed', 'Radar-KF speed', 'DETECTION']
    rows = ('NoInjection', 'PositionInjection', 'SpeedInjection', 'AccelerationInjection', 'AllInjection', 'CoordinatedInjection')
   # data = [[ 66386, 174296,  75131, 577908,  32015, 174296,  75131, 577908,  32015]]
    ax2.axis('tight')
    ax2.axis('off')
    tab = ax2.table(cellText=summary_detect, colWidths=[0.13 for x in columns], rowLabels=rows, colLabels=columns, loc="center",  bbox=[0.02,0.05,1.1,1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)

    plt.show()
