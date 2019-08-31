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
from AutoTensor.data_mgmt.data_loader import load_dataset_file, prepare_dataset


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


def find_optimal_model(
    X: pd.DataFrame, y: pd.Series, val_ratio: float, test_ratio: float
) -> None:
    # TODO: Support any type that the pd.DataFrame and pd.Series
    # constructors can take, just by passing X and y through those
    # constructors.
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


if __name__ == "__main__":
    args = get_cli_args()
    X, y = load_dataset_file(args.path, args.target_index)
    find_optimal_model(X, y, args.val_ratio, args.test_ratio)
