from __future__ import absolute_import, division, print_function
import json

import tensorflow as tf
import pprint

from AutoTensor.q_learning.QLearner import QLearner
from AutoTensor.q_learning.tensor_q_mapper import actions, starting_state, compose_get_state, compose_get_reward
from AutoTensor.data_mgmt.data_loader import get_my_arff_dataset, load_json_str


def find_optimal_model(get_dataset):
    data, num_classes = get_dataset()

    get_state = compose_get_state()
    get_reward = compose_get_reward(data, num_classes)

    # acc = get_reward(starting_state)
    # print(
    #     "======== FINISHED WITH TEST OF CURRYING AND TENSOR FLOW - ACCURACY IS {} ========"
    #     .format(acc))

    qlearner = QLearner(
        discount=0.9,
        alpha=0.2,
        get_state=get_state,
        get_reward=get_reward,
        starting_state=starting_state,
        actions=actions,
        report_file_path="AutoTensor/reports/q-learner-results.json")

    qlearner.find_state_with_best_q()


if __name__ == "__main__":
    find_optimal_model(get_my_arff_dataset)