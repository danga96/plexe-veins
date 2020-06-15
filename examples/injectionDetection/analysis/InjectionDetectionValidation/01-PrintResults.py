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
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExtractValidationResults:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path, low_memory=False, converters={
            'run': ExtractValidationResults.__parse_run_column,
            'attrvalue': ExtractValidationResults.__parse_attrvalue_column,
            'vectime': ExtractValidationResults.__parse_ndarray,
            'vecvalue': ExtractValidationResults.__parse_ndarray})

        self.attack_start = ExtractValidationResults.__extract_attack_start(self.data)
        self.crashed = ExtractValidationResults.__extract_crashed(self.data)
        self.leader_detection, self.predecessor_detection =\
            ExtractValidationResults.__extract_attack_detections(self.data)

    def print_summary(self, verbose):

        def typeToIneq(type):
            return int((np.log2(type) + 2))

        _runs, _ = self.leader_detection.shape

        _leader_detected = self.leader_detection[self.leader_detection["LeaderAttackDetected"] != -1]
        _predecessor_detected = self.predecessor_detection[self.predecessor_detection["PredecessorAttackDetected"] != -1]

        _leader_detected_join = self.attack_start.join(_leader_detected)
        _predecessor_detected_join = self.attack_start.join(_predecessor_detected)

        def _remove_negative(ds):
            return ds[ds > 0]

        _leader_detected_delay = -1 if len(_leader_detected_join) == 0 else\
            _remove_negative(_leader_detected_join.apply((lambda _row: _row["LeaderAttackDetected"] - _row["AttackStart"]), axis=1)).mean()
        _predecessor_detected_delay = -1 if len(_predecessor_detected_join) == 0 else\
            _remove_negative(_predecessor_detected_join.apply((lambda _row: _row["PredecessorAttackDetected"] - _row["AttackStart"]), axis=1)).mean()

        print("Leader attack detected:     ", len(_leader_detected), "out of", _runs, "({:.2f}%)".format(100 * len(_leader_detected)/_runs), "after", "{:.2f}".format(_leader_detected_delay), "seconds")
        print("Predecessor attack detected:", len(_predecessor_detected), "out of", _runs, "({:.2f}%)".format(100 * len(_predecessor_detected)/_runs), "after",  "{:.2f}".format(_predecessor_detected_delay), "seconds")
        print("Crashes:                    ", self.crashed.sum(), "out of", _runs, "({:.2f}%)".format(100 * self.crashed.sum()/_runs))

        if verbose:
            if len(_leader_detected > 0):
                print()
                print("Leader attack detected type:")
                print(_leader_detected["LeaderAttackDetectedType"].apply(typeToIneq).value_counts().to_string())

            if len(_predecessor_detected > 0):
                print()
                print("Predecessor attack detected type:")
                print(_predecessor_detected["PredecessorAttackDetectedType"].apply(typeToIneq).value_counts().to_string())

    @staticmethod
    def __extract_attack_start(data):
        _idx_attacker = 0
        _filter_attack_start = np.logical_and(
              np.logical_and(data.type == "scalar", data.name.str.contains("AttackStart")),
              np.logical_and(np.logical_not(data.module.isnull()), data.module.str.contains(str(_idx_attacker)))
        )

        return data[_filter_attack_start].pivot("run", columns="name", values="value")

    @staticmethod
    def __extract_crashed(data):
        _crashed = data[np.logical_and(data.type == "scalar", data.name == "Crashed")]\
            .groupby("run").agg({"value": np.sum})
        _crashed = _crashed.assign(crashed=_crashed.value.apply(lambda value: value != 0))
        return _crashed["crashed"]

    @staticmethod
    def __extract_attack_detections(data):
        _idx_predecessor_detect, _idx_leader_detect = 1, 7
        _filter_leader_detection = np.logical_and(
              np.logical_and(data.type == "scalar", data.name.str.contains("LeaderAttackDetected")),
              np.logical_and(np.logical_not(data.module.isnull()), data.module.str.contains(str(_idx_leader_detect)))
        )
        _filter_predecessor_detection = np.logical_and(
              np.logical_and(data.type == "scalar", data.name.str.contains("PredecessorAttackDetected")),
              np.logical_and(np.logical_not(data.module.isnull()), data.module.str.contains(str(_idx_predecessor_detect)))
        )

        _leader_detection = data[_filter_leader_detection].pivot("run", columns="name", values="value")
        _predecessor_detection = data[_filter_predecessor_detection].pivot("run", columns="name", values="value")

        return _leader_detection, _predecessor_detection

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
    parser.add_argument("inputs", nargs="+", help="Files to be processed")
    parser.add_argument("--verbose", action="store_true", help="Print a more verbose summary")
    args = parser.parse_args()

    for _input in args.inputs:
        print("Processing file '%s'..." % _input)

        _base_path, _filename = os.path.split(_input)
        extractor = ExtractValidationResults(_base_path, _filename)
        extractor.print_summary(args.verbose)

        print("Finished processing file '%s'...\n" % _input)
