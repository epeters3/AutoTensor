import json

from AutoTensor.Action import Action
from AutoTensor.tensor_runner import evaluate_model

action_map = {}
actions = action_map.keys()

starting_state = ""


def compose_get_state():
    def get_state(state, action):
        pass

    return get_state


def compose_get_reward(train_data, train_labels, val_data, val_labels,
                       test_data, test_labels):
    def get_reward(state):
        return evaluate_model(state, train_data, train_labels, val_data,
                              val_labels, test_data, test_labels, 100, 5)

    return get_reward