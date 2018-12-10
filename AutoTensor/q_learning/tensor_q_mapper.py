import json
import pprint

import tensorflow as tf
from tensorflow import keras

from AutoTensor.q_learning.Action import Action
from AutoTensor.tensorflow.tensor_runner import evaluate_model
from AutoTensor.q_learning.config_scheme import SchemeManager
from AutoTensor.q_learning.action_builder import ActionBuilder
from AutoTensor.q_learning.config_builder import ConfigBuilder

action_builder = ActionBuilder()
action_map = action_builder.build(SchemeManager.scheme)

# actions is exported and consumed
actions = action_map.keys()
actions.sort()

print("Built actions successfully.\nWill use these actions:")
pprint.pprint(actions)

config_builder = ConfigBuilder()

# starting_state is also exported and consumed
starting_state = config_builder.build(SchemeManager.scheme)
print("Config built successfully.\nWill use starting_state:")
pprint.pprint(starting_state)


def compose_get_state():
    def get_state(state, action_name):
        new_state = action_map[action_name].do(state)
        return new_state

    return get_state


def compose_get_reward(data, num_classes):
    def get_reward(state):
        reward = evaluate_model(state, data, num_classes, 0)
        return reward

    return get_reward