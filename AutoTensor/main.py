import json
from argparse import ArgumentParser

import tensorflow as tf
import pprint
import pandas as pd

from AutoTensor.q_learning.QLearner import QLearner
from AutoTensor.q_learning.tensor_q_mapper import (
    actions,
    starting_state,
    compose_get_state,
    compose_get_reward,
)
from AutoTensor.data_mgmt.data_loader import load_dataset_file
from AutoTensor.data_mgmt.dataset import prepare_dataset


def get_cli_args():
    parser = ArgumentParser(
        description="Conduct Neural Architecture Search on a dataset file."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="The path to the dataset file to train on.",
    )
    parser.add_argument(
        "--target-index",
        "-ti",
        type=int,
        default=-1,
        help=(
            "The 0-based index of the target column in the dataset file. "
            "Negative indexing is also supported. If not supplied, it is "
            "assumed the last column is the target."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        "-vr",
        type=float,
        default=0.15,
        help="The ratio of the dataset to use for the validation set.",
    )
    parser.add_argument(
        "--test-ratio",
        "-tr",
        type=float,
        default=0.15,
        help="The ratio of the dataset to use for the holdout test set.",
    )
    return parser.parse_args()


def _find_optimal_model(
    X: pd.DataFrame, y: pd.Series, val_ratio: float, test_ratio: float
) -> None:
    dataset = prepare_dataset(X, y, val_ratio, test_ratio)

    get_state = compose_get_state()
    get_reward = compose_get_reward(dataset)

    qlearner = QLearner(
        discount=0.9,
        alpha=0.2,
        get_state=get_state,
        get_reward=get_reward,
        starting_state=starting_state,
        actions=actions,
        report_file_path="AutoTensor/reports/q-learner-results.json",
    )

    qlearner.find_state_with_best_q()


def find_optimal_model(X, y, val_ratio: float, test_ratio: float) -> None:
    """
    Public facing wrapper function for `_find_optimal_model`. Any values for
    `X and `y` that a pandas DataFrame or Series constructor can take are valid.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    _find_optimal_model(X, y, val_ratio, test_ratio)


if __name__ == "__main__":
    args = get_cli_args()
    X, y = load_dataset_file(args.path, args.target_index)
    _find_optimal_model(X, y, args.val_ratio, args.test_ratio)
