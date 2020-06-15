import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    def __init__(self, base_path, file_name, use_prediction=False):

        _path = os.path.join(base_path, file_name)
        print(_path)
        self.data = pd.read_csv(_path, converters={
            'run': InjectionDetectionAnalyzer.__parse_run_column,
            'attrvalue': InjectionDetectionAnalyzer.__parse_attrvalue_column,
            'vectime': InjectionDetectionAnalyzer.__parse_ndarray,
            'vecvalue': InjectionDetectionAnalyzer.__parse_ndarray})
        
        def _get_sensor_error(attrname, default=0.):
            try:
                return float(self.data[self.data["attrname"] == attrname]["attrvalue"])
            except TypeError:
                return default

        self.sensor_params = {
            "ego-gps": _get_sensor_error("*.node[*].sensors[1..2].absoluteError", default=1),
            "ego-speed": _get_sensor_error("*.node[*].sensors[3..5].absoluteError", default=0.1),
            "ego-acceleration": _get_sensor_error("*.node[*].sensors[6].absoluteError", default=0.01),
            "radar-position": _get_sensor_error("*.node[*].sensors[8].absoluteError", default=0.1),
            "radar-speed": _get_sensor_error("*.node[*].sensors[9].absoluteError", default=0.1),
        }

        self.vehicles = 8

        self.sampling_times = []
        self.vehicle_data = []

        self.kf_predictions = []
        self.kf_predictions_speed_only = []

        self.leader_attack_detected = np.zeros(self.vehicles)
        self.predecessor_attack_detected = np.zeros(self.vehicles)
        #print( self.data.loc[self.data.name == "V2XTime"])
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

            _positions = _current_vehicle_data.loc["V2XPositionX"]["vecvalue"][:_length]
            _speeds = _current_vehicle_data.loc["V2XSpeed"]["vecvalue"][:_length]
            _accelerations = _current_vehicle_data.loc["V2XAcceleration"]["vecvalue"][:_length]

            _dt = (self.sampling_times[_i + 1] - self.sampling_times[_i])\
                if _i < len(self.sampling_times) - 1 and use_prediction else 0
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

        self.cacc_params = {
            "spacing": float(self.data[self.data["attrname"] == "spacing"]["attrvalue"]),
            "headway": float(self.data[self.data["attrname"] == "headway"]["attrvalue"]),
            "vehicle-length": 4,  # meters
        }

        try:
            self.attack_start = float(self.data[self.data["attrname"] == "AttackStart"]["attrvalue"])
        except TypeError:
            self.attack_start = None

        self.default_window_size = 10

    def plot_comparison_graph(self, title):
        """
        Plots a graph showing a comparison between the values obtained through V2X
        and the ones estimated by the Kalman filter
        """

        _f, _ax = plt.subplots(4, 1, num=title, sharex="all", gridspec_kw={"height_ratios": [2, 1, 2, 1]},
                               squeeze=False)

        _vehicle = 0

        _sampling_times = self.sampling_times[_vehicle]
        _advertised = self.vehicle_data[_vehicle]["V2XPosition"]
        _expected = self.kf_predictions[_vehicle]["kfEstimatedPosition"]
        _variance = self.kf_predictions[_vehicle]["kfEstimatedPositionVar"]
        _measurement_sigma = np.repeat(1, len(_sampling_times))
        _running_avg = InjectionDetectionAnalyzer.__running_avg(_advertised - _expected, self.default_window_size)

        _ax.flat[0].plot(_sampling_times, _advertised, label="Advertised")
        _ax.flat[0].plot(_sampling_times, _expected, label="Estimated")
        _ax.flat[1].plot(_sampling_times, _advertised - _expected, lw=1)
        _ax.flat[1].plot(_sampling_times, _running_avg)
        _ax.flat[1].plot(_sampling_times, np.sqrt(_variance) * 3, "--", color="C3")
        _ax.flat[1].plot(_sampling_times, -np.sqrt(_variance) * 3, "--", color="C3")
        _ax.flat[1].plot(_sampling_times, _measurement_sigma * 3, ":", color="C4")
        _ax.flat[1].plot(_sampling_times, -_measurement_sigma * 3, ":", color="C4")

        _advertised = self.vehicle_data[_vehicle]["V2XSpeed"]
        _expected = self.kf_predictions[_vehicle]["kfEstimatedSpeed"]
        _variance = self.kf_predictions[_vehicle]["kfEstimatedSpeedVar"]
        _measurement_sigma = np.repeat(0.05, len(_sampling_times))
        _running_avg = InjectionDetectionAnalyzer.__running_avg(_advertised - _expected, self.default_window_size)
        _ax.flat[2].plot(_sampling_times, _advertised, label="Advertised")
        _ax.flat[2].plot(_sampling_times, _expected, label="Estimated")
        _ax.flat[3].plot(_sampling_times, _advertised - _expected, lw=1)
        _ax.flat[3].plot(_sampling_times, _running_avg)
        _ax.flat[3].plot(_sampling_times, np.sqrt(_variance) * 3, "--", color="C3")
        _ax.flat[3].plot(_sampling_times, -np.sqrt(_variance) * 3, "--", color="C3")
        _ax.flat[3].plot(_sampling_times, _measurement_sigma * 3, ":", color="C4")
        _ax.flat[3].plot(_sampling_times, -_measurement_sigma * 3, ":", color="C4")

        _ax.flat[0].set_xlabel("Time (s)")
        _ax.flat[0].set_ylabel("Position (m)")
        _ax.flat[0].legend()
        _ax.flat[1].set_xlabel("Time (s)")
        _ax.flat[1].set_ylabel("Delta Position")
        _ax.flat[2].set_xlabel("Time (s)")
        _ax.flat[2].set_ylabel("Speed (m/s)")
        _ax.flat[2].legend()
        _ax.flat[3].set_xlabel("Time (s)")
        _ax.flat[3].set_ylabel("Delta Speed")

    def plot_distance_graph_all(self, title, ax=None, window_size=None, legend=True):
        """
        Plots a graph showing the pairwise distance between two vehicles
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if title is not None:
            ax.set_title(title)

        if window_size is None:
            window_size = self.default_window_size

        _lines = []
        for _i in range(len(self.vehicle_data) - 1):
            _first_vec = self.vehicle_data[_i]["V2XPositionComp"]
            _second_vec = self.vehicle_data[_i + 1]["V2XPosition"]

            _time = self.sampling_times[_i]
            _expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(
                self.cacc_params, self.vehicle_data[_i + 1]["V2XSpeed"])

            _first_vec_avg = InjectionDetectionAnalyzer.__running_avg(_first_vec, window_size)
            _second_vec_avg = InjectionDetectionAnalyzer.__running_avg(_second_vec, window_size)

            _difference = _first_vec - _second_vec - _expected_distance - self.cacc_params["vehicle-length"]
            _difference_avg = _first_vec_avg - _second_vec_avg - _expected_distance - self.cacc_params["vehicle-length"]
            _lines.append(ax.plot(_time, _difference_avg, label="{} - {}".format(_i + 1, _i) if legend else None))

        _threshold = 5
        ax.axhline(_threshold, color="C4", linestyle="--")
        ax.axhline(-_threshold, color="C4", linestyle="--")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Distance (m)")

        if self.attack_start:
            ax.axvline(self.attack_start, color="red", linestyle=":", lw=1)

        return _lines

    def plot_distance_graph(self, title, ax=None, show_attack_detected=False, legend=True):
        """
        Plots a graph showing the distance between two vehicles, computed both considering the GPS readings and the
        values computed by mean of the Kalman filter
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        _vehicle = 0
        _sampling_times = self.sampling_times[_vehicle]

        _expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(
            self.cacc_params, self.vehicle_data[_vehicle + 1]["V2XSpeed"])

        _gps_distance = self.vehicle_data[_vehicle]["V2XPositionComp"] - \
                        self.vehicle_data[_vehicle + 1]["V2XPosition"] - \
                        _expected_distance - self.cacc_params["vehicle-length"]

        _kf_distance = self.kf_predictions[_vehicle]["kfEstimatedPositionComp"] - \
                       self.kf_predictions[_vehicle + 1]["kfEstimatedPosition"] - \
                       _expected_distance - self.cacc_params["vehicle-length"]

        _kf_distance_avg = InjectionDetectionAnalyzer.__running_avg(_gps_distance - _kf_distance, self.default_window_size)

        _kf_distance_speed = self.kf_predictions_speed_only[_vehicle]["kfEstimatedPositionComp"] - \
                             self.kf_predictions_speed_only[_vehicle + 1]["kfEstimatedPosition"]

        # Compute the threshold as a percentage of the expected distance
        _threshold_distance = _expected_distance * 0.33
        _threshold_error = (np.sqrt(self.kf_predictions[_vehicle]["kfEstimatedPositionVarComp"]) +
                            np.sqrt(self.kf_predictions[_vehicle + 1]["kfEstimatedPositionVar"])) * 3

        _lines = [
            # ax.plot(_sampling_times, _gps_distance, lw=1, label="V2X Distance" if legend else None),
            ax.plot(_sampling_times, _gps_distance, label="V2X Distance AVG" if legend else None),
            ax.plot(_sampling_times, _kf_distance, label="KF Prediction" if legend else None),

            ax.plot(_sampling_times, _kf_distance_avg, label="(V2X - KF) AVG" if legend else None),
            # ax.plot(_sampling_times, _kf_distance - _kf_distance_speed, label="KF (GPS + speed) - KF (speed only)" if legend else None),
        ]

        ax.plot(_sampling_times, _threshold_error, color="C7", linestyle="--")
        ax.plot(_sampling_times, -_threshold_error, color="C7", linestyle="--")
        ax.plot(_sampling_times, _threshold_distance, color="C8", linestyle="--")
        ax.plot(_sampling_times, -_threshold_distance, color="C8", linestyle="--")

        if self.attack_start:
            ax.axvline(self.attack_start, color="red", linestyle="--", lw=1)

        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.leader_attack_detected, color="green")
        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.predecessor_attack_detected, color="orange")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Distance (m)")

        if show_attack_detected:
            _regions = InjectionDetectionAnalyzer.__compute_attack_region(
                _threshold_distance, _sampling_times, _kf_distance)
            _id1 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax, color="yellow")

            _regions = InjectionDetectionAnalyzer.__compute_attack_region(
                _threshold_error, _sampling_times, _kf_distance_avg)
            _id2 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax)

            _title_ad = InjectionDetectionAnalyzer.__title_attack_detection(self.attack_start, _id1, _id2)
            title = "{} ({})".format(title, _title_ad)

        ax.set_title(title)
        return _lines

    def plot_radar_distance_graph(self, title, ax=None, legend=True, show_attack_detected=False):
        """
        Plots a graph showing the pairwise distance between two vehicles (compared with the ones obtained through
        the radar).
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        _vehicle = 0

        _expected_distance = InjectionDetectionAnalyzer.__compute_expected_distance(
            self.cacc_params, self.vehicle_data[_vehicle + 1]["V2XSpeed"])

        _sampling_times = self.sampling_times[_vehicle]
        _v2x_distance = self.vehicle_data[_vehicle]["V2XPositionComp"] - \
                        self.vehicle_data[_vehicle + 1]["V2XPosition"] - \
                        _expected_distance - self.cacc_params["vehicle-length"]
        _v2x_distance_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_distance, self.default_window_size)
        _kf_distance = self.kf_predictions[_vehicle]["kfEstimatedPositionComp"] - \
                       self.kf_predictions[_vehicle + 1]["kfEstimatedPosition"] - \
                       _expected_distance - self.cacc_params["vehicle-length"]
        _radar_distance = self.vehicle_data[_vehicle + 1]["RadarDistance"] - _expected_distance
        _radar_distance_avg = InjectionDetectionAnalyzer.__running_avg(_radar_distance, self.default_window_size)

        _lines = [
            # ax.plot(_sampling_times, _v2x_distance, lw=1, label="V2X Distance" if legend else None),
            ax.plot(_sampling_times, _v2x_distance_avg, label="V2X Distance AVG" if legend else None),
            ax.plot(_sampling_times, _kf_distance, label="KF Distance" if legend else None),
            # ax.plot(_sampling_times, _radar_distance, lw=1, label="Radar distance" if legend else None),
            ax.plot(_sampling_times, _radar_distance_avg, label="Radar distance AVG" if legend else None),
            ax.plot(_sampling_times, _v2x_distance_avg - _radar_distance_avg, label="V2X AVG - Radar AVG" if legend else None),
            ax.plot(_sampling_times, _kf_distance - _radar_distance_avg, label="KF - Radar AVG" if legend else None),
        ]

        # Compute the threshold as a percentage of the expected distance
        _threshold_distance = _expected_distance * 0.25
        _threshold_error = _threshold_distance

        ax.plot(_sampling_times, _threshold_error, color="C7", linestyle="--")
        ax.plot(_sampling_times, -_threshold_error, color="C7", linestyle="--")
        ax.plot(_sampling_times, _threshold_distance, color="C8", linestyle="--")
        ax.plot(_sampling_times, -_threshold_distance, color="C8", linestyle="--")

        if self.attack_start:
            ax.axvline(self.attack_start, color="red", linestyle="--", lw=1)

        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.leader_attack_detected, color="green")
        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.predecessor_attack_detected, color="orange")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Distance (m)")

        if show_attack_detected:
            _regions = InjectionDetectionAnalyzer.__compute_attack_region(
                _threshold_distance, _sampling_times, _kf_distance - _radar_distance_avg)
            _id1 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax)

            _regions = InjectionDetectionAnalyzer.__compute_attack_region(
                _threshold_distance, _sampling_times, _radar_distance_avg)
            _id2 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax, color="yellow")

            _title_ad = InjectionDetectionAnalyzer.__title_attack_detection(self.attack_start, _id1, _id2)
            title = "{} ({})".format(title, _title_ad)

        ax.set_title(title)
        return _lines

    def plot_speed_difference_graph(self, title, ax=None, window_sizes=None, legend=True,
                                    show_attack_detected=False, attack_window_size_idx=0):
        """
        Plots a graph showing the difference between the speed values obtained through V2X and the ones estimated by the
        Kalman filter, along with a configurable number of running means.
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if window_sizes is None:
            window_sizes = [self.default_window_size, ]

        _vehicle = 0

        _sampling_times = self.sampling_times[_vehicle]
        _v2x_speed = self.vehicle_data[_vehicle]["V2XSpeed"]
        _kf_speed = self.kf_predictions[_vehicle]["kfEstimatedSpeed"]
        _kf_speed_std = np.sqrt(self.kf_predictions[_vehicle]["kfEstimatedSpeedVar"])

        _v2x_accel = np.abs(self.vehicle_data[_vehicle + 1]["V2XAcceleration"])
        _v2x_accel_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_accel, 25)
        _threshold = (self.sensor_params["ego-speed"] + _kf_speed_std * 3) * (1 + 0.05 * _v2x_accel)

        _lines = [
            ax.plot(_sampling_times, _v2x_speed - _kf_speed, lw=1, label="V2X - KF" if legend else None),
        ]

        _running_avgs = dict()
        for _window_size in window_sizes:
            _avg_difference = InjectionDetectionAnalyzer.__running_avg(_v2x_speed - _kf_speed, _window_size)
            _running_avgs[_window_size] = _avg_difference
            _lines.append(ax.plot(_sampling_times, _avg_difference,
                                  label="Running avg: {}".format(_window_size) if legend else None))

        ax.plot(_sampling_times, _threshold, color="C8", linestyle="--")
        ax.plot(_sampling_times, -_threshold, color="C8", linestyle="--")

        if self.attack_start:
            ax.axvline(self.attack_start, color="red", linestyle="--", lw=1)

        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.leader_attack_detected, color="green")
        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.predecessor_attack_detected, color="orange")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Delta Speed (m/s)")

        if show_attack_detected:
            _running_avg = _running_avgs[window_sizes[attack_window_size_idx]]
            _regions = InjectionDetectionAnalyzer.__compute_attack_region(_threshold, _sampling_times, _running_avg)
            _id = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax)

            _title_ad = InjectionDetectionAnalyzer.__title_attack_detection(self.attack_start, _id)
            title = "{} ({})".format(title, _title_ad)

        ax.set_title(title)
        return _lines

    def plot_radar_speed_difference_graph(self, title, ax=None, legend=True, show_attack_detected=False):
        """
        Plots a graph showing the difference between the speed values estimated by the Kalman filter
        and the radar readings.
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        _vehicle = 0

        _sampling_times = self.sampling_times[_vehicle]
        _v2x_speed_diff = self.vehicle_data[_vehicle]["V2XSpeedComp"] - self.vehicle_data[_vehicle + 1]["V2XSpeed"]
        _v2x_speed_diff_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_speed_diff, self.default_window_size)
        _kf_speed_diff = self.kf_predictions[_vehicle]["kfEstimatedSpeedComp"] - \
                         self.kf_predictions[_vehicle + 1]["kfEstimatedSpeed"]
        _radar_speed = self.vehicle_data[_vehicle + 1]["RadarRelativeSpeed"]
        _radar_speed_avg = InjectionDetectionAnalyzer.__running_avg(_radar_speed, self.default_window_size)

        _radar_v2x_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_speed_diff - _radar_speed,
                                                                  self.default_window_size)
        _radar_kf_avg = InjectionDetectionAnalyzer.__running_avg(_kf_speed_diff - _radar_speed, self.default_window_size)
        _v2x_kf_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_speed_diff - _kf_speed_diff,
                                                               self.default_window_size)

        _lines = [
            # ax.plot(_sampling_times, _v2x_speed_diff, lw=1, label="V2X Difference" if legend else None),
            ax.plot(_sampling_times, _v2x_speed_diff_avg, label="V2X Difference AVG" if legend else None),
            ax.plot(_sampling_times, _kf_speed_diff, label="KF Difference" if legend else None),
            # ax.plot(_sampling_times, _radar_speed, lw=1, label="Radar measurement" if legend else None),
            ax.plot(_sampling_times, _radar_speed_avg, label="Radar measurement AVG" if legend else None),
            ax.plot(_sampling_times, _radar_v2x_avg, label="(V2X - Radar) AVG" if legend else None),
            ax.plot(_sampling_times, _radar_kf_avg, label="(KF - Radar) AVG" if legend else None),
        ]

        _v2x_accel = np.abs(self.vehicle_data[_vehicle + 1]["V2XAcceleration"])
        _v2x_accel_avg = InjectionDetectionAnalyzer.__running_avg(_v2x_accel, 25)
        _threshold = (self.sensor_params["ego-speed"] + self.sensor_params["radar-speed"]) * (1 + 0.05 * _v2x_accel_avg)

        ax.plot(_sampling_times, _threshold, color="C8", linestyle="--")
        ax.plot(_sampling_times, -_threshold, color="C8", linestyle="--")

        if self.attack_start:
            ax.axvline(self.attack_start, color="red", linestyle="--", lw=1)

        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.leader_attack_detected, color="green")
        InjectionDetectionAnalyzer.__plot_attack_detected(ax, self.predecessor_attack_detected, color="orange")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Relative Speed (m/s)")

        if show_attack_detected:
            _regions = InjectionDetectionAnalyzer.__compute_attack_region(_threshold, _sampling_times, _radar_v2x_avg)
            _id1 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax)

            _regions = InjectionDetectionAnalyzer.__compute_attack_region(_threshold, _sampling_times, _radar_kf_avg)
            _id2 = InjectionDetectionAnalyzer.__plot_attack_region(_regions, ax, color="yellow")

            _title_ad = InjectionDetectionAnalyzer.__title_attack_detection(self.attack_start, _id1, _id2)
            title = "{} ({})".format(title, _title_ad)

        ax.set_title(title)
        return _lines

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
    def __compute_attack_region(threshold, sampling_times, *values):
        _attack_detected = False
        for _array in values:
            _attack_detected = np.logical_or(_attack_detected, np.logical_or(_array > threshold, _array < -threshold))

        # Skip the first few seconds to avoid setup problems
        _attack_detected[:50] = False

        _attack_detected = _attack_detected[:-1] != _attack_detected[1:]
        _attack_detected_time = sampling_times[:-1][_attack_detected]
        if len(_attack_detected_time) % 2 != 0:
            _attack_detected_time = np.append(_attack_detected_time, sampling_times[-1])
        return _attack_detected_time

    @staticmethod
    def __plot_attack_region(attack_regions, ax, color="red", min_duration=1.):
        _first_detection = None
        for _min, _max in zip(attack_regions[::2], attack_regions[1::2]):
            if _max - _min >= min_duration:
                ax.axvspan(_min + min_duration, _max, color=color, alpha=0.25)
                if _first_detection is None:
                    _first_detection = _min + min_duration
        return _first_detection
                
    @staticmethod
    def __title_attack_detection(attack_start, *detections):
        _invalid = 1e6
        detections = np.array(detections)
        #print(detections[detections != np.array(None)])
        _first_detection = np.min(detections[detections != np.array(None)], initial=_invalid)

        return "No attack detected" if _first_detection == _invalid else \
            "Attack erroneously detected" if attack_start is None else \
            "Attack detected after {:.1f}s".format(_first_detection - attack_start)

    @staticmethod
    def __plot_attack_detected(ax, attack_detected, color):
        try:
            _detected = np.min(attack_detected[attack_detected > 0.])
            ax.axvline(_detected, color=color, linestyle="--", lw=1)
        except ValueError:
            pass

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
        for _legend_line, _line in zip(legend.get_lines(), lines):
            _legend_line.set_picker(5)  # 5 pts tolerance
            _lined[_legend_line] = _line

        def _on_line_pick(event):
            _fn_legend_line = event.artist
            _fn_lines = _lined[_fn_legend_line]

            _visible = False
            for _fn_line in _fn_lines:
                _fn_line.set_visible(not _fn_line.get_visible())
                _visible = _fn_line.get_visible()

            # Change the alpha on the line in the legend so we can see what lines have been toggled
            _fn_legend_line.set_alpha(1.0 if _visible else 0.5)
            plt.draw()

        figure.canvas.mpl_connect('pick_event', _on_line_pick)


if __name__ == "__main__":

    base_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/summary/"
    scenario = "Random"
    controller = "CACC"

    #base_path = os.path.join(base_path, controller)
    analyzers = {
        "NoInjection": InjectionDetectionAnalyzer(base_path, "{}NoInjection.csv".format(scenario)),
        "PositionInjection": InjectionDetectionAnalyzer(base_path, "{}PositionInjection.csv".format(scenario)),
        "SpeedInjection": InjectionDetectionAnalyzer(base_path, "{}SpeedInjection.csv".format(scenario)),
        "AccelerationInjection": InjectionDetectionAnalyzer(base_path, "{}AccelerationInjection.csv".format(scenario)),
        "AllInjection": InjectionDetectionAnalyzer(base_path, "{}AllInjection.csv".format(scenario)),
        "CoordinatedInjection": InjectionDetectionAnalyzer(base_path, "{}CoordinatedInjection.csv".format(scenario))
    }

    f1, ax1 = plt.subplots(len(analyzers), 1, sharex="all", num="{} Scenario - {} - Radar Distance".format(scenario, controller))
    f2, ax2 = plt.subplots(len(analyzers), 1, sharex="all", num="{} Scenario - {} - Radar Speed".format(scenario, controller))
    f3, ax3 = plt.subplots(len(analyzers), 1, sharex="all", num="{} Scenario - {} - Distance".format(scenario, controller))
    f4, ax4 = plt.subplots(len(analyzers), 1, sharex="all", num="{} Scenario - {} - Speed".format(scenario, controller))

    f1.suptitle("Distance between leader and second vehicle (Radar)")
    f2.suptitle("Speed difference between leader and second vehicle (Radar)")
    f3.suptitle("Distance between leader and second vehicle")
    f4.suptitle("Leader speed evaluation with Kalman Filters")

    f1_lines, f2_lines, f3_lines, f4_lines, f5_lines = [], [], [], [], []
    for i, key in enumerate(analyzers):
        args = {"show_attack_detected": True, "legend": i == 0}
        args2 = {"window_sizes": [5, 10, 25, 50], "attack_window_size_idx": 1}
        f1_lines.append(analyzers[key].plot_radar_distance_graph(key, ax1[i], **args))
        f2_lines.append(analyzers[key].plot_radar_speed_difference_graph(key, ax2[i], **args))
        f3_lines.append(analyzers[key].plot_distance_graph(key, ax3[i], **args))
        f4_lines.append(analyzers[key].plot_speed_difference_graph(key, ax4[i], **args, **args2))

    InjectionDetectionAnalyzer.setup_hide_lines(f1, np.array(f1_lines).T[0], f1.legend())
    InjectionDetectionAnalyzer.setup_hide_lines(f2, np.array(f2_lines).T[0], f2.legend())
    InjectionDetectionAnalyzer.setup_hide_lines(f3, np.array(f3_lines).T[0], f3.legend())
    InjectionDetectionAnalyzer.setup_hide_lines(f4, np.array(f4_lines).T[0], f4.legend())

    plt.show()
