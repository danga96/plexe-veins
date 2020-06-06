import argparse
import os
import re

import numpy as np
import pandas as pd


class FeaturesExtraction:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path, converters={
            'run': FeaturesExtraction.__parse_run_column,
            'attrvalue': FeaturesExtraction.__parse_attrvalue_column,
            'vectime': FeaturesExtraction.__parse_ndarray,
            'vecvalue': FeaturesExtraction.__parse_ndarray})

        self.data.attrvalue.replace("CACC", "PATH", inplace=True)
        self.data.attrvalue.replace("CONSENSUS", "Consensus", inplace=True)
        self.data.attrvalue.replace("FLATBED", "Flatbed", inplace=True)
        self.data.attrvalue.replace("PLOEG", "Ploeg", inplace=True)

        self.parameters = FeaturesExtraction.__extract_parameters(self.data)
        self.vectors = FeaturesExtraction.__extract_vectors(self.data)
        self.crashed = FeaturesExtraction.__extract_crashed(self.data)
        self.distances = FeaturesExtraction.__compute_min_max_distances(self.parameters, self.vectors, self.crashed)

        assert np.all(self.distances["crashed"] == self.distances["crashed_comp"])

    def save_to_csv(self, base_path, filename):
        _path = os.path.join(base_path, filename)
        self.distances.to_csv(_path)

    @staticmethod
    def __extract_parameters(data):
        _parameters = data[data.type.isin(["runattr", "param"])] \
            .drop_duplicates(subset=["run", "attrname"], keep="last") \
            .pivot("run", columns="attrname", values="attrvalue") \
            .rename(lambda col: col.split(".")[-1], axis="columns")  # Extract just the final part of the column name

        # Remove the duplicated columns
        return _parameters.loc[:, ~_parameters.columns.duplicated()]

    @staticmethod
    def __extract_vectors(data):
        _vectors = data[np.logical_and(data.type == "vector", data.module.str.contains("appl"))]
        _vectors = _vectors.assign(
            vehicle=_vectors.module.apply(lambda value: 1 + int(re.search('.*([0-9]+)', value).group(1)))
        )

        _vector_columns = ["run", "vehicle", "name", "vectime", "vecvalue"]
        return _vectors[_vector_columns]

    @staticmethod
    def __extract_crashed(data):
        _crashed = data[data.type == "scalar"].groupby("run").agg({"value": np.sum})
        _crashed = _crashed.assign(crashed_comp=_crashed.value.apply(lambda val: val != 0))
        return _crashed["crashed_comp"]

    @staticmethod
    def __compute_min_max_distances(parameters, vectors, crashed):

        def __limit_less_than_zero(values):
            values[values < 0] = 0
            return values

        _distances = vectors[np.logical_and(vectors.name == "distance", vectors.vehicle > 1)]
        _distances = _distances.assign(
            min_distance=__limit_less_than_zero(_distances.vecvalue.apply(np.min)),
            max_distance=_distances.vecvalue.apply(np.max)
        )
        _distances.drop(["vectime", "vecvalue"], axis=1, inplace=True)

        # For each run, extract the row with the minimum distance (with vehicle number)
        _crashed_vehicles = _distances.loc[_distances.groupby("run")["min_distance"].idxmin()].set_index("run")

        _distances = _distances.groupby("run").agg({
            "min_distance": np.min,
            "max_distance": np.max
        })

        _distances = parameters.join(_distances)
        _distances = _distances.assign(expected_distance=_distances.apply(
            FeaturesExtraction.__compute_expected_distance, axis=1
        ))

        _distances = _distances.assign(
            min_distance_delta=_distances.apply(lambda row: row.expected_distance - row.min_distance, axis=1),
            max_distance_delta=_distances.apply(lambda row: row.max_distance - row.expected_distance, axis=1),
        )

        _index_columns = ["controller", "spacing", "headway", "leaderSpeed", "useRadarPredSpeed", "attackStart", "repetition"]
        _columns = _index_columns + ["min_distance", "expected_distance", "max_distance", "min_distance_delta", "max_distance_delta"]
        _distances = _distances[_columns].set_index(_index_columns)

        _speeds = vectors[vectors.name == "speed"]
        _crash_speeds = _speeds.assign(crash_speed=_speeds.vecvalue.apply(lambda array: array[-1]))
        _crash_speeds = _crash_speeds[["run", "vehicle", "crash_speed"]].set_index("run")

        _crashed_vehicles = _crashed_vehicles.assign(crashed=_crashed_vehicles.min_distance == 0)
        _crashed_vehicles = parameters.join(_crashed_vehicles).join(crashed)
        _columns = _index_columns + ["vehicle", "crashed", "crashed_comp"]
        _crashed_vehicles = _crashed_vehicles[_columns].set_index(_index_columns)

        return _distances.join(_crashed_vehicles)

    @staticmethod
    def __compute_expected_distance(row):
        return row.platoonInsertSpeed * row.headway / 3.6 + row.spacing

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
