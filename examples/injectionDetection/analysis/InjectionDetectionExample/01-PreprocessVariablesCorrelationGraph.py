import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KalmanFilter import KalmanFilter


class FeaturesExtraction:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path, converters={
            'run': FeaturesExtraction.__parse_run_column,
            'attrvalue': FeaturesExtraction.__parse_attrvalue_column,
            'vectime': FeaturesExtraction.__parse_ndarray,
            'vecvalue': FeaturesExtraction.__parse_ndarray})

        self.vehicles = 8
        self.__extract_sensor_parameters()
        self.__extract_vehicle_data()
        self.__compute_injected_kf(25, 1.0/3.6, 10.0/3.6)

    def save_to_csv(self, base_path, filename):
        filename = filename.replace("NoAttack", "VariablesCorrelation")
        _path = os.path.join(base_path, filename)
        pd.DataFrame.from_dict(self.vehicle_data)[["Time", "V2XSpeed", "V2XSpeedInj", "KFSpeed", "KFSpeedInj"]]\
            .set_index("Time").to_csv(_path)

    def __extract_sensor_parameters(self):

        def _extract_sensor_parameter(attrname, default=0.):
            try:
                return float(self.data[self.data["attrname"] == attrname]["attrvalue"])
            except TypeError:
                return default

        self.sensor_params = {
            "ego-gps": _extract_sensor_parameter("*.node[*].sensors[1..2].absoluteError", default=1),
            "ego-speed": _extract_sensor_parameter("*.node[*].sensors[3..5].absoluteError", default=0.1),
            "ego-acceleration": _extract_sensor_parameter("*.node[*].sensors[6].absoluteError", default=0.01),
            "radar-distance": _extract_sensor_parameter("*.node[*].sensors[8].absoluteError", default=0.1),
            "radar-speed": _extract_sensor_parameter("*.node[*].sensors[9].absoluteError", default=0.1),
        }

    def __extract_vehicle_data(self, vehicle_idx=0):

        _filter = np.logical_and(
            np.logical_and(np.logical_not(self.data.module.isnull()), self.data.module.str.contains(str(vehicle_idx))),
            self.data.attrname.isnull()
        )
        _vehicle_data = self.data[_filter][["name", "vecvalue"]].set_index("name")

        _positions = _vehicle_data.loc["V2XPositionX"]["vecvalue"]
        _speeds = _vehicle_data.loc["V2XSpeed"]["vecvalue"]
        _accelerations = _vehicle_data.loc["V2XAcceleration"]["vecvalue"]

        _vars = [_positions, _speeds, _accelerations, self.sensor_params]
        _kf_pred, _kf_var = FeaturesExtraction.__apply_kalman_filter(*_vars)

        self.vehicle_data = {
            "Time": _vehicle_data.loc["V2XTime"]["vecvalue"],

            "V2XPosition": _positions,
            "V2XSpeed": _speeds,
            "V2XAcceleration": _accelerations,

            "KFPosition": _kf_pred[:, 0],
            "KFSpeed": _kf_pred[:, 1],
        }

    def __compute_injected_kf(self, attack_start, attack_rate, attack_limit):
        _times, _injected_speeds = self.vehicle_data["Time"], np.copy(self.vehicle_data["V2XSpeed"])
        for _i, _time in enumerate(_times):
            if _time > attack_start:
                _injected_speeds[_i] += min(attack_rate*(_time - attack_start), attack_limit)

        _vars = [self.vehicle_data["V2XPosition"], _injected_speeds, self.vehicle_data["V2XAcceleration"]]
        _kf_pred, _ = FeaturesExtraction.__apply_kalman_filter(*_vars, self.sensor_params)

        self.vehicle_data["V2XSpeedInj"] = _injected_speeds
        self.vehicle_data["KFPositionInj"] = _kf_pred[:, 0]
        self.vehicle_data["KFSpeedInj"] = _kf_pred[:, 1]

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
    def __parse_run_column(value):
        match = re.search('([a-zA-Z]+)-([0-9]+)-(.*)', value)

        # Configuration, Run, Timestamp
        return match.group(1), int(match.group(2)), match.group(3)

    @staticmethod
    def __parse_attrvalue_column(value):

        _float_regex = '([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        _unit_regex = '\\s*(?:[a-zA-Z]+)?'

        match = re.fullmatch(_float_regex + _unit_regex, value)
        return float(match.group(1)) if match else \
            True if value == "true" else False if value == "false" else value.replace('"', '') if value else None

    @staticmethod
    def __parse_ndarray(value):
        return np.fromstring(value, sep=' ') if value else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="The path where the output files are saved")
    parser.add_argument("inputs", nargs="+", help="Files to be processed")
    args = parser.parse_args()

    for _input in args.inputs:
        print("Processing file '%s'..." % _input)

        _base_path, _filename = os.path.split(_input)
        features_extraction = FeaturesExtraction(_base_path, _filename)
        features_extraction.save_to_csv(args.output_path, _filename)

        print("Finished processing file '%s'...\n" % _input)
