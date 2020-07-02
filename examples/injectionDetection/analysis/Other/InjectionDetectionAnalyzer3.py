import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

summary_detect = np.zeros( (6, 8) )
summary_best_th = []

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

    def __init__(self, data, detection_parameters, simulation_index, use_prediction=False):

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

        try:
            self.attack_start = float(self.data[self.data["attrname"] == "attackStart"]["attrvalue"])
        except TypeError:
            self.attack_start = None

    def detection_analyzer(self):

        try:
            self.attack_start = float(self.data[self.data["name"] == "attackStart"]["value"])
        except TypeError:
            self.attack_start = None
        #    print("DOPO" , self.attack_start)

        #if self.attack_start_vector[0]:
        #    ax.axvline(self.attack_start_vector[0], color="black", linestyle=":", lw=1)

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
    
        # KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": _kf_distance - self._expected_distance - self.cacc_params["vehicle-length"],
            "thresholds": self.detection_parameters["distanceKFThresholdFactor"] * self._expected_distance,
        }
        self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 1, **_data)
        print("KF distance",_data["thresholds"][-1])
        """
        # V2X Distance - KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_v2x_distance - _kf_distance, _window),
            "thresholds": 3 * self._kf_position_std * self.detection_parameters["distanceV2XKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C2", "V2X-KF distance" if legend else None, **_data))
        #self._optimal_threshold(self.detection_parameters["attackTolerance"], self.attack_start_vector[0], 2, **_data)
        print("V2X-KF distance",_data["thresholds"][-1])
        
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
            "values": _foll_data["RadarDistance"] - _expected_distance,
            "thresholds": self.detection_parameters["distanceRadarThresholdFactor"] * _expected_distance,
        }
        _lines.append(self._plot_detection_line(ax, "C4", "Radar distance" if legend else None, **_data))
        print("Radar distance",_data["thresholds"][-1])
        
        # Radar Distance - KF Distance
        _data = {
            "sampling_times": _sampling_times,
            "values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance + self.cacc_params["vehicle-length"], _window),
            #"values": InjectionDetectionAnalyzer.__running_avg(_foll_data["RadarDistance"] - _kf_distance, _window),
            "thresholds": (self.sensor_params["radar-distance"] + 3 * _kf_position_std) *
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
            "thresholds": (self.sensor_params["radar-speed"] + _kf_speed_std * 3) *
                          (1 + np.abs(_pred_data["V2XAcceleration"]) * self.detection_parameters["accelerationFactor"]) *
                           self.detection_parameters["speedRadarKFThresholdFactor"],
        }
        _lines.append(self._plot_detection_line(ax, "C7", "Radar-KF speed" if legend else None, **_data))
        print("Radar-KF speed",_data["thresholds"][-1])
        """
        # Attack detected
        #ax.axvline(self.predecessor_attack_detected[1], color="red", linewidth=1, label="Detection" if legend else None)

        #summary_detect[self.simulation][7] = round(self.predecessor_attack_detected[1],4)

        summary_detect[0][0] = round(self.attack_start_vector[0],4)

        return None

    def _optimal_threshold(self, attack_tolerance, attack_start, _eq, sampling_times, values, thresholds ):
        _attack_detected = np.abs(values) > thresholds
        #ciclare fino a quando da attack start ci sono 10 true, evtando di triggerarne 10 prima di attacck start -> optimal policy
        #estrarre il 10 dopo attack_start
        _index = 0
        print("sample",len(sampling_times)," values",len(values))
        rd = attack_start%0.1
        print("rd",rd, " atk st",attack_start)
        for _i, _v in enumerate(sampling_times):
            if _v > attack_start:
                _index = _i
                break

        print("index",_index," value",sampling_times[_index])
        #TODO controllare che _index non sfori
        #_index += 7
        _best_detect = 100#TODO forse non serve
        _best_th = 1#TODO con orignale, invece di 1
        _attack_detected_original = InjectionDetectionAnalyzer.__compute_attack_start(sampling_times, values, thresholds, attack_tolerance)
        if _attack_detected_original is None:
            _attack_detected_original = 100
        for _i in range(0,len(sampling_times)):
            #print("threshold",thresholds[_i]," values",values[_i]," sampl",sampling_times[_i]," kf_pos_std",_kf_position_std[_i])
            th_test = self._get_alpha_test(_eq,_i,values)
            #print("new_th",th_test)
            thresholds = self._set_th(_eq,th_test)
            _attack_detected = InjectionDetectionAnalyzer.__compute_attack_start(sampling_times, values, thresholds, attack_tolerance)
            #print("det:",_attack_detected," nth:",th_test, " i:",_i)
            if _attack_detected is None:
                _attack_detected = 0
            if _attack_detected>attack_start and _attack_detected<_attack_detected_original:
                #se attack_detected_orignal è 100, vuol dire che sono riuscito a rilevare un attacco che con la vecchia soglia non veniva rilevato
                if _attack_detected<_best_detect or th_test<_best_th:
                    _best_detect = _attack_detected
                    _best_th = th_test
                    #print("new_th_test",th_test)
                #break
        
        print("best detect ",_best_detect, " best th:",_best_th)
        #print("simulation index",self.simulation_index, "   eq",_eq)
        summary_best_th[self.simulation_index+1][_eq-1] = round(_best_th,4)
        return None

    def _get_alpha_test(self, _eq, _i, values):
        switcher = {
            1: abs(values[_i]/self._expected_distance[_i]),
            2: abs(values[_i]/(3*self._kf_position_std[_i])),
        }
        return switcher.get(_eq, 0)

    def _set_th(self, _eq, th_test):
        switcher = {
            1: th_test * self._expected_distance,
            2: th_test * 3 * self._kf_position_std,
        }
        return switcher.get(_eq, 0)
                

    @staticmethod
    def __compensate_position(dt, position, speed, acceleration):
        return position + dt * speed + 0.5 * dt * dt * acceleration

    @staticmethod
    def __compensate_speed(dt, speed, acceleration):
        return speed + dt * acceleration

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
    
    AllAttacks = ["{}PositionInjection.csv".format(scenario), "{}AllInjection.csv".format(scenario)]
    for _attack_index, attack in enumerate(AllAttacks):
        data_object = CollectDataForAttack(base_path, attack)
        test_data = data_object.get_data()
        grouped = test_data.groupby("run")

        summary_best_th = np.zeros( (len(test_data.run.unique())+1, 7) )
        summary_best_th[0] = [detection_parameters["distanceKFThresholdFactor"], 
                                detection_parameters["distanceV2XKFThresholdFactor"],0,0,0,0,0]
       
        
        row_sim = []
        for simulation_index, simulation in enumerate(test_data.run.unique()):#per ogni simulazione
            data = grouped.get_group(simulation)
            analyzer = InjectionDetectionAnalyzer(data, detection_parameters, simulation_index)
            analyzer.detection_analyzer()
            row_sim.append("Sim{}".format(simulation_index))
            simulation_index += 1
            print("-----------------------------------------------------------------------------------------------------------",simulation)

        print(summary_best_th)
        f1, ax1 = plt.subplots(2,1)
        f1.suptitle(attack)
        f1.set_figwidth(12)
        columns = ['KF distance', 'V2X-KF distance', 'V2X-KF speed', 'Radar distance', 'Radar-KF distance', 'Radar-V2X speed', 'Radar-KF speed']
        rows = ["Original"]
        rows += row_sim
        #ax1.axis('tight')
        ax1[0].axis('off')
        #ax1.autoscale(enable=True, axis="Both")
              
        tab = ax1[0].table(cellText=summary_best_th, rowLabels=rows, colLabels=columns, loc="center")
        tab.auto_set_font_size(False)
        tab.auto_set_column_width([0,1,2,3,4,5,6,7])
        #tab.scale(1,1.5)
        tab.set_fontsize(11)

        ax1[1].axis('off')
        columns += ["Start|Pdetect"]
        rows.remove("Original")
        #ax1.autoscale(enable=True, axis="Both")   
        summary_best_th = np.zeros( (len(test_data.run.unique()), 8) )       
        tab2 = ax1[1].table(cellText=summary_best_th, rowLabels=rows, colLabels=columns, loc="center")
        tab2.auto_set_font_size(False)
        tab2.auto_set_column_width([0,1,2,3,4,5,6,7])
        #tab.scale(1,1.5)
        tab2.set_fontsize(11)

        f1.tight_layout()#adapt table in width
        
    plt.show()
    exit()
    analyzers = {
        #"NoInjection": InjectionDetectionAnalyzer(base_path, "{}NoInjection.csv".format(scenario), detection_parameters),
        #"PositionInjection": InjectionDetectionAnalyzer(base_path, "{}PositionInjection.csv".format(scenario), detection_parameters),
        #"SpeedInjection": InjectionDetectionAnalyzer(base_path, "{}SpeedInjection.csv".format(scenario), detection_parameters),
        #"AccelerationInjection": InjectionDetectionAnalyzer(base_path, "{}AccelerationInjection.csv".format(scenario), detection_parameters),
        "AllInjection": InjectionDetectionAnalyzer(base_path, "{}AllInjection.csv".format(scenario), detection_parameters),
        #"CoordinatedInjection": InjectionDetectionAnalyzer(base_path, "{}CoordinatedInjection.csv".format(scenario), detection_parameters)
    }

   