import json
from argparse import ArgumentParser

import tensorflow as tf
import pprint

from AutoTensor.q_learning.QLearner import QLearner
from AutoTensor.q_learning.tensor_q_mapper import (
    actions,
    starting_state,
    compose_get_state,
    compose_get_reward,
)
from AutoTensor.data_mgmt.data_loader import load_dataset_file


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
        "-t",
        type=int,
        default=-1,
        help=(
            "The 0-based index of the target column in the dataset file. "
            "Negative indexing is also supported. If not supplied, it is "
            "assumed the last column is the target."
        ),
    )
    return parser.parse_args()


def find_optimal_model(data, num_classes):

    get_state = compose_get_state()
    get_reward = compose_get_reward(data, num_classes)

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
    data, num_classes = load_dataset_file(args.path, args.target_index)
    find_optimal_model(data, num_classes)
