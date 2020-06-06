import argparse
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


class SummaryTablePrinter:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path)

    def get_summary(self, name, radar):
        def _return_summary(row):
            _min, _exp = row.min_distance_delta, row.expected_distance
            return 0 if _min < 0.1 * _exp else \
                   1 if _min < 0.5 * _exp else \
                   2 if _min < 0.75 * _exp else 3

        _filtered_data = self.data[self.data.useRadarPredSpeed == radar]

        _column = {name: _filtered_data.apply(_return_summary, axis=1)}
        _filtered_data = _filtered_data.assign(**_column)
        return _filtered_data[["controller", name]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", help="The path where the input files are stored")
    parser.add_argument("scenario", help="The selected scenario (e.g. Constant or Sinusoidal)")
    args = parser.parse_args()

    summaries = {}
    attack_types = ["PositionInjection", "SpeedInjection", "AccelerationInjection", "AllInjection"]

    for attack_type in attack_types:
        summaries[attack_type] = SummaryTablePrinter(args.base_path, "{}{}.csv".format(args.scenario, attack_type))

    summary = summaries[attack_types[0]].get_summary("Position", False)
    summary = summary.merge(summaries[attack_types[1]].get_summary("Speed", False))
    summary = summary.merge(summaries[attack_types[1]].get_summary("Speed Radar", True))
    summary = summary.merge(summaries[attack_types[2]].get_summary("Acceleration", False))

    print(summary)
