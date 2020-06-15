import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlotSpeedProfile:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path, converters={
            'run': PlotSpeedProfile.__parse_run_column,
            'attrvalue': PlotSpeedProfile.__parse_attrvalue_column,
            'vectime': PlotSpeedProfile.__parse_ndarray,
            'vecvalue': PlotSpeedProfile.__parse_ndarray})

        self.speed_profiles = PlotSpeedProfile.__extract_speed_profiles(self.data)
        self.acceleration_profiles = PlotSpeedProfile.__extract_acceleration_profiles(self.data)

    def plot_speed_profiles(self, nrows):

        _df_rows, _ = self.speed_profiles.shape
        _ncols = np.ceil(_df_rows / nrows).astype(int)
        _f, _ax_matrix = plt.subplots(nrows, _ncols, sharex="all", sharey="all", squeeze=False)

        for _ax, _row in zip(_ax_matrix.flat, self.speed_profiles.iterrows()):
            _ax.plot(_row[1]["vectime"], _row[1]["vecvalue"] * 3.6)
        _f.tight_layout()

    def plot_acceleration_profiles(self, nrows):

        _df_rows, _ = self.acceleration_profiles.shape
        _ncols = np.ceil(_df_rows / nrows).astype(int)
        _f, _ax_matrix = plt.subplots(nrows, _ncols, sharex="all", sharey="all", squeeze=False)

        for _ax, _row in zip(_ax_matrix.flat, self.acceleration_profiles.iterrows()):
            _ax.plot(_row[1]["vectime"], _row[1]["vecvalue"])
        _f.tight_layout()

    @staticmethod
    def __extract_speed_profiles(data):

        _leader_vehicle_id = 0
        _filter = np.logical_and(
            data.name == "speed",
            np.logical_and(np.logical_not(data.module.isnull()), data.module.str.contains(str(_leader_vehicle_id)))
        )

        _speed_profiles = data[_filter]
        return _speed_profiles.assign(
            runid=_speed_profiles.run.apply(lambda r: r[1])
        ).set_index("runid")[["vectime", "vecvalue"]]

    @staticmethod
    def __extract_acceleration_profiles(data):

        _leader_vehicle_id = 0
        _filter = np.logical_and(
            data.name == "acceleration",
            np.logical_and(np.logical_not(data.module.isnull()), data.module.str.contains(str(_leader_vehicle_id)))
        )

        _acceleration_profiles = data[_filter]
        return _acceleration_profiles.assign(
            runid=_acceleration_profiles.run.apply(lambda r: r[1])
        ).set_index("runid")[["vectime", "vecvalue"]]

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
    parser.add_argument("input", help="The input file to be plot")
    parser.add_argument("--acceleration", action="store_true", help="Print the acceleration profile instead of the speed one")
    args = parser.parse_args()

    _base_path, _filename = os.path.split(args.input)
    plotter = PlotSpeedProfile(_base_path, _filename)
    if not args.acceleration:
        plotter.plot_speed_profiles(nrows=1)
    else:
        plotter.plot_acceleration_profiles(nrows=1)
    plt.show()
