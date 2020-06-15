import argparse
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


class VariablesCorrelationGraphPlotter:

    def __init__(self, base_path, file_name):

        _path = os.path.join(base_path, file_name)
        self.data = pd.read_csv(_path)

    def plot_graph(self):

        plt.plot(self.data["Time"], self.data["V2XSpeed"], label="V2XSpeed")
        plt.plot(self.data["Time"], self.data["V2XSpeedInj"], label="V2XSpeedInj")
        plt.plot(self.data["Time"], self.data["KFSpeed"], label="KFSpeed")
        plt.plot(self.data["Time"], self.data["KFSpeedInj"], label="KFSpeedInj")
        plt.legend()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The input file to be plot")
    args = parser.parse_args()

    _base_path, _filename = os.path.split(args.input)
    plotter = VariablesCorrelationGraphPlotter(_base_path, _filename)
    plotter.plot_graph()
    plt.show()
