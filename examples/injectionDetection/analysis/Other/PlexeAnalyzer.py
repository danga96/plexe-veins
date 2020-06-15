import re
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


class PlexeAnalyzer:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path, converters={
            'run': PlexeAnalyzer.__parse_run_column,
            'attrvalue': PlexeAnalyzer.__parse_attrvalue_column,
            'vectime': PlexeAnalyzer.__parse_ndarray,
            'vecvalue': PlexeAnalyzer.__parse_ndarray})

        self.parameters = PlexeAnalyzer.__extract_parameters(self.data)
        self.vectors = PlexeAnalyzer.__extract_vectors(self.data)
        self.distances, self.crashed_vehicles, self.attack_start_values = \
            PlexeAnalyzer.__compute_min_max_distances(self.parameters, self.vectors)

        # Backward compatibility
        if "spacing" not in self.parameters.columns:
            self.parameters = self.parameters.assign(spacing=self.parameters.platoonInsertDistance)
        if "headway" not in self.parameters.columns:
            self.parameters = self.parameters.assign(spacing=self.parameters.platoonInsertHeadway)

    def plot_distances_summary_graph(self, ax=None, attack_start=None, **kwargs):
        if not ax:
            ax = plt.figure().add_subplot(111)

        if attack_start is not None and attack_start not in self.attack_start_values:
            sys.stderr.write("Invalid attack start_value: {}".format(attack_start))
            return

        _distances = self.distances[self.distances.index.get_level_values("attackStart") == attack_start] \
            if attack_start is not None else self.distances

        _name = _distances.apply(
            lambda _row: '{0} ({1:.1f}m + {2:.1f}s)'.format(*_row.name),
            axis=1).tolist()
        _exp_dist = _distances.expected_distance.values
        _limit_dist = _distances[["min_distance", "max_distance"]].values.transpose()
        _err_dist = np.abs(_exp_dist - _limit_dist)

        ax.errorbar(y=_name, x=_exp_dist, xerr=_err_dist, fmt="|", markersize=6, **kwargs)
        ax.invert_yaxis()

    def plot_variable_graph(self, title, variable_name="distance", attack_start=None, sharey="none"):

        def _format_label(row):
            return '{0} ({1:.1f}m + {2:.1f}s)'.format(row.controller[0], row.spacing[0], row.headway[0])

        _variables_ylabel = {
            "distance": "Distance (m)",
            "speed": "Speed (m/s)",
            "acceleration": "Acceleration (m/s2)",
            "controllerAcceleration": "Controller acceleration (m/s2)"
        }

        if variable_name not in _variables_ylabel:
            sys.stderr.write("Invalid variable_name value: {}\n".format(variable_name))
            return

        if attack_start is not None and attack_start not in self.attack_start_values:
            sys.stderr.write("Invalid attack_start value: {}\n".format(attack_start))
            return

        _attack_start_condition = self.parameters["attackStart"] == attack_start \
            if attack_start is not None else [True, ] * len(self.parameters)
        _skip_leader_condition = self.vectors.vehicle > 0 \
            if variable_name == "distance" else True

        _runs = self.parameters[_attack_start_condition].index
        _values = self.vectors[np.logical_and(self.vectors.name == variable_name, _skip_leader_condition)]

        _f, _ax = plt.subplots(len(_runs), 1, sharex="all", sharey=sharey, num=title, squeeze=False)
        _f.suptitle(title)

        for _i, _run in enumerate(_runs):
            _ax.flat[_i].set_title(_format_label(self.parameters.loc[[_run]]))

            # Workaround to force the same line colors as in the other graphs (with one additional line)
            if variable_name == "distance":
                _ax.flat[_i].plot([], [])

            for _row in _values[_values.run == _run].itertuples():
                _ax.flat[_i].plot(_row.vectime, _row.vecvalue, label=_row.vehicle if _i == 0 else None)

        _f.legend(loc="right")
        _f.tight_layout()

        # Workaround to display common label
        _f.text(0.5, 0.04, "Time (s)", ha='center')
        _f.text(0.04, 0.5, _variables_ylabel[variable_name], va='center', rotation='vertical')

    @staticmethod
    def __extract_parameters(data):
        _parameters = data[data.type.isin(["attr", "param"])] \
            .drop_duplicates(subset=["run", "attrname"], keep="last") \
            .pivot("run", columns="attrname", values="attrvalue") \
            .rename(lambda col: col.split(".")[-1], axis="columns")  # Extract just the final part of the column name

        # Remove the duplicated columns
        return _parameters.loc[:, ~_parameters.columns.duplicated()]

    @staticmethod
    def __extract_vectors(data):
        _vectors = data[np.logical_and(data.type == "vector", data.module.str.contains("appl"))]
        _vectors = _vectors.assign(
            vehicle=_vectors.module.apply(lambda value: int(re.search('.*([0-9]+)', value).group(1)))
        )

        _vector_columns = ["run", "vehicle", "name", "vectime", "vecvalue"]
        return _vectors[_vector_columns]

    @staticmethod
    def __compute_min_max_distances(parameters, vectors):

        def __limit_less_than_zero(values):
            values[values < 0] = 0
            return values

        _distances = vectors[np.logical_and(vectors.name == "distance", vectors.vehicle > 0)]
        _distances = _distances.assign(
            min_distance=__limit_less_than_zero(_distances.vecvalue.apply(np.min)),
            max_distance=_distances.vecvalue.apply(np.max)
        )

        # For each run, extract the row with the minimum distance (with vehicle number)
        _crashed_vehicles = _distances.loc[_distances.groupby("run")["min_distance"].idxmin()].set_index("run")

        _distances = _distances.groupby("run").agg({
            "min_distance": np.min,
            "max_distance": np.max
        })

        _distances = parameters.join(_distances)
        _distances = _distances.assign(expected_distance=_distances.apply(
            PlexeAnalyzer.__compute_expected_distance, axis=1
        ))

        _index_columns = ["controller", "spacing", "headway", "leaderSpeed"]
        _attack_start_values = [None]
        if "attackStart" in _distances.columns:
            _attack_start_values = sorted(_distances["attackStart"].unique().tolist())
            _index_columns.append("attackStart")

        _columns = _index_columns + ["min_distance", "expected_distance", "max_distance"]
        _distances = _distances[_columns].set_index(_index_columns)

        _crashed_vehicles = _crashed_vehicles.assign(crashed=_crashed_vehicles.min_distance == 0)
        _crashed_vehicles = parameters.join(_crashed_vehicles)
        _index_columns.append("vehicle")
        _columns = _index_columns + ["crashed", "min_distance"]
        _crashed_vehicles = _crashed_vehicles[_columns].set_index(_index_columns)

        return _distances, _crashed_vehicles, _attack_start_values

    @staticmethod
    def __compute_expected_distance(row):
        return row.platoonInsertSpeed * row.headway / 3.6 + row.spacing

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


def _plot_reference():
    sinusoidal = PlexeAnalyzer("../JammingAttack", "{}NoAttack.csv".format("Sinusoidal"))
    configurable = PlexeAnalyzer("../JammingAttack", "{}NoAttack.csv".format("Configurable"))

    fig, _ax = plt.subplots(2, 2, sharex="all", sharey="none", num="Reference")

    x, y = sinusoidal.vectors[sinusoidal.vectors.name == "speed"].iloc[0][["vectime", "vecvalue"]]
    _ax.flat[0].set_title("Sinusoidal")
    _ax.flat[0].set_xlabel("Time (s)")
    _ax.flat[0].set_ylabel("Speed (km/h)")
    _ax.flat[0].plot(x, y*3.6)

    x, y = sinusoidal.vectors[sinusoidal.vectors.name == "acceleration"].iloc[0][["vectime", "vecvalue"]]
    _ax.flat[1].set_title("Sinusoidal")
    _ax.flat[1].set_xlabel("Time (s)")
    _ax.flat[1].set_ylabel("Acceleration (m/s2)")
    _ax.flat[1].plot(x, y)

    for value in [30, 31, 32, 32.5, 33, 34]:
        _ax.flat[0].axvline(value, color="red", linestyle=":")
        _ax.flat[1].axvline(value, color="red", linestyle=":")

    x, y = configurable.vectors[configurable.vectors.name == "speed"].iloc[0][["vectime", "vecvalue"]]
    _ax.flat[2].set_title("Configurable")
    _ax.flat[2].set_xlabel("Time (s)")
    _ax.flat[2].set_ylabel("Speed (km/h)")
    _ax.flat[2].plot(x, y*3.6)

    x, y = configurable.vectors[configurable.vectors.name == "controllerAcceleration"].iloc[0][["vectime", "vecvalue"]]
    _ax.flat[3].set_title("Configurable")
    _ax.flat[3].set_xlabel("Time (s)")
    _ax.flat[3].set_ylabel("Acceleration (m/s2)")
    _ax.flat[3].plot(x, y)

    for value in [7.5, 12.5, 17.5, 27.5, 40, 48]:
        _ax.flat[2].axvline(value, color="red", linestyle=":")
        _ax.flat[3].axvline(value, color="red", linestyle=":")


def _plot_jamming_attack(base_path, scenario, plot_variable=None, attack_start_idx=0):
    no_attack = PlexeAnalyzer(base_path, "{}NoAttack.csv".format(scenario))
    leader_jammed = PlexeAnalyzer(base_path, "{}LeaderJammed.csv".format(scenario))
    all_jammed = PlexeAnalyzer(base_path, "{}AllJammed.csv".format(scenario))
    all_jammed_radar = PlexeAnalyzer(base_path, "{}AllJammedRadar.csv".format(scenario))
    all_jammed_prediction = PlexeAnalyzer(base_path, "{}AllJammedPrediction.csv".format(scenario))

    attack_start_values = leader_jammed.attack_start_values

    if plot_variable is not None:
        params = {
            "variable_name": plot_variable,
            "attack_start": attack_start_values[attack_start_idx]
        }

        no_attack.plot_variable_graph(title="No Attack", variable_name=plot_variable)
        leader_jammed.plot_variable_graph(title="Leader Jammed", **params)
        all_jammed.plot_variable_graph(title="All Jammed", **params)
        all_jammed_radar.plot_variable_graph(title="All Jammed Radar", **params)
        all_jammed_prediction.plot_variable_graph(title="All Jammed Prediction", **params)

    nx = int(np.ceil(len(attack_start_values)/2))
    fig, _ax = plt.subplots(nx, 2, sharex="none", sharey="none", num="{} Summary".format(scenario), squeeze=False)
    fig.suptitle("{} Summary".format(scenario))
    fig.text(0.5, 0.04, "Distance from the preceding vehicle (m)", ha='center')

    for i, attack_start in enumerate(attack_start_values):
        _ax.flat[i].set_title("Attack start: {0}s".format(attack_start))
        _ax.flat[i].axvline(0, color="red", linestyle=":")

        offset = transforms.ScaledTranslation(0, -3 / 72, fig.dpi_scale_trans)
        transform = _ax.flat[i].transData - offset - offset
        params = {
            "ax": _ax.flat[i],
            "lw": 2.5,
            "capsize": 2,
            "transform": transform,
        }

        no_attack.plot_distances_summary_graph(color="C0", label="No Attack", **params)

        params["attack_start"] = attack_start
        params["transform"] += offset
        leader_jammed.plot_distances_summary_graph(color="C1", label="Leader Jammed", **params)

        params["transform"] += offset
        all_jammed.plot_distances_summary_graph(color="C2", label="All Jammed", **params)

        params["transform"] += offset
        all_jammed_radar.plot_distances_summary_graph(color="C3", label="All Jammed Radar", **params)

        params["transform"] += offset
        #  all_jammed_prediction.plot_distances_summary_graph(color="C4", label="All Jammed Prediction", **params)

        _ax.flat[i].margins(0.10)
        _ax.flat[i].legend(loc="best")


def _plot_injection_attack(base_paths, scenario):

    fig, _ax = plt.subplots(1, len(base_paths), sharex="none", sharey="none", num="{} Summary".format(scenario))
    fig.suptitle("{} Summary".format(scenario))
    fig.text(0.5, 0.04, "Distance from the preceding vehicle (m)", ha='center')

    for i, base_path in enumerate(base_paths):

        # no_attack = PlexeAnalyzer(base_path, "{}NoAttack.csv".format(scenario))
        position_injection = PlexeAnalyzer(base_path, "{}PositionInjection.csv".format(scenario))
        speed_injection = PlexeAnalyzer(base_path, "{}SpeedInjection.csv".format(scenario))
        speed_injection_radar = PlexeAnalyzer(base_path, "{}SpeedInjectionRadar.csv".format(scenario))
        acceleration_injection = PlexeAnalyzer(base_path, "{}AccelerationInjection.csv".format(scenario))
        time_injection = PlexeAnalyzer(base_path, "{}TimeInjection.csv".format(scenario))
        all_injection = PlexeAnalyzer(base_path, "{}AllInjection.csv".format(scenario))
        all_injection_radar = PlexeAnalyzer(base_path, "{}AllInjectionRadar.csv".format(scenario))

        offset = transforms.ScaledTranslation(0, -5 / 72, fig.dpi_scale_trans)
        transform = _ax.flat[i].transData - offset - offset - offset

        params = {
            "ax": _ax.flat[i],
            "lw": 3.5,
            "capsize": 2,
            "transform": transform,
        }

        # no_attack.plot_distances_summary_graph(_ax, color="C0", label="No Attack", **params)

        params["transform"] += offset
        position_injection.plot_distances_summary_graph(color="C1", label="Position Injection", **params)

        params["transform"] += offset
        speed_injection.plot_distances_summary_graph(color="C2", label="Speed Injection", **params)

        params["transform"] += offset
        speed_injection_radar.plot_distances_summary_graph(color="C3", label="Speed Injection Radar", **params)

        params["transform"] += offset
        acceleration_injection.plot_distances_summary_graph(color="C4", label="Acceleration Injection", **params)

        params["transform"] += offset
        time_injection.plot_distances_summary_graph(color="C5", label="Time Injection", **params)

        params["transform"] += offset
        all_injection.plot_distances_summary_graph(color="C6", label="All Injection", **params)

        params["transform"] += offset
        all_injection_radar.plot_distances_summary_graph(color="C7", label="All Injection Radar", **params)

        _ax.flat[i].set_title(base_path)
        _ax.flat[i].axvline(0, color="red", linestyle=":")
        _ax.flat[i].margins(0.10)
        _ax.flat[i].legend(loc="best")


if __name__ == "__main__":
    # _plot_reference()

    # _plot_jamming_attack("../PlexeData/JammingAttack", "Sinusoidal")
    # _plot_jamming_attack("../PlexeData/JammingAttack", "Configurable")

    _plot_injection_attack(["../PlexeData/InjectionAttack", "../PlexeData/InjectionAttackLeader"], "Constant")
    # _plot_injection_attack(["../PlexeData/InjectionAttack", "../PlexeData/InjectionAttackLeader"], "RandomConstant")

    plt.show()
