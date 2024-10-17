"""
This script depends on:
* pytorch
* numpy (1.2.x)
* sklearn
* matplotlib
"""

import numpy as np
import torch
import argparse

from numpy import ndarray
from torch import Tensor

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Stores the BNN architecture and training logic
from fut_network import *


def train_and_save():
    model = BayesianNN()


def load_bnn():
    pass


def connect():
    pass


def get_args(arg_override=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "BNN Exploration",
        "python fut [mode]",
        description="If no mode is provided, mode defaults to --full.",
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--full",
        "-f",
        action="store_true",
        default=False,
        help="Generates a new model, trains, saves, and explores the solution space.",
    )

    mode.add_argument(
        "--train",
        "-t",
        action="store_true",
        default=False,
        help="Only generates a new model, trains, and saves it.",
    )

    mode.add_argument(
        "--explore",
        "-x",
        action="store_true",
        default=False,
        help=(
            "Only explores an existing model under --model-path. If no model exists "
            "under said path, the program will fail."
        ),
    ),

    parser.add_argument(
        "--model-path",
        "-p",
        type=str,
        default="models/bnn_expl/",
        help="Where to store and load the BNN model.",
    )

    parser.add_argument(
        "--model-name",
        "-n",
        type=str,
        default="bnn.model",
        help="The name of the bnn model file.",
    )

    return parser.parse_args(arg_override)


def main(arg_override=None):
    import os

    args = get_args(arg_override)

    pre_trained_exists = os.path.isfile(f"{args.model_path}/{args.model_name}")

    print(f"Pre-trained model exists: {pre_trained_exists}")


if __name__ == "__main__":
    main()
