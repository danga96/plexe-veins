# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2019 Marco Iorio <marco.iorio@polito.it>

import argparse
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


class SummaryGraphPlotter:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path)

    def plot_summary_graph(self, ax, radar, **kwargs):

        _filtered_data = self.data[self.data.useRadarPredSpeed == radar]

        _name = _filtered_data.controller.tolist()
        _exp_dist = _filtered_data.expected_distance.values
        _err_dist = _filtered_data[["min_distance_delta", "max_distance_delta"]].values.transpose()

        ax.errorbar(y=_name, x=_exp_dist, xerr=_err_dist, fmt="|", markersize=6, **kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", help="The path where the input files are stored")
    parser.add_argument("scenario", help="The selected scenario (e.g. Constant or Sinusoidal)")
    args = parser.parse_args()

    f, ax = plt.subplots(1, 1)

    plotters = {}
    attack_types = ["PositionInjection", "SpeedInjection", "AccelerationInjection", "AllInjection"]

    offset = transforms.ScaledTranslation(0, -5 / 72, f.dpi_scale_trans)
    transform = ax.transData - offset - offset - offset
    params = {"ax": ax, "lw": 3.5, "capsize": 2, "transform": transform}
    i = 0

    for attack_type in attack_types:
        plotters[attack_type] = SummaryGraphPlotter(args.base_path, "{}{}.csv".format(args.scenario, attack_type))

        plotters[attack_type].plot_summary_graph(color="C{}".format(i), label=attack_type, radar=False, **params)
        i += 1
        params["transform"] += offset

        if attack_type in ["SpeedInjection", "AllInjection"]:
            plotters[attack_type].plot_summary_graph(color="C{}".format(i), label="{} (Radar)".format(attack_type), radar=True, **params)
            i += 1
            params["transform"] += offset

    ax.invert_yaxis()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Controller")
    f.legend()
    plt.show()
