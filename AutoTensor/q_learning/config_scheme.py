from __future__ import division
import tensorflow as tf
from tensorflow import keras


class ClassNode:
    def __init__(self, name, class_ref, args):
        self.name = name
        self.class_ref = class_ref
        self.args = args


class OptionsNode:
    def __init__(self, name, options):
        self.name = name
        self.options = options
        self.default = 0


class ValueNode:
    actions = {
        "increment": lambda x: max(0, x + 1),
        "decrement": lambda x: max(0, x - 1),
        "increment5": lambda x: max(0, x + 5),
        "decrement5": lambda x: max(0, x - 5),
        "double": lambda x: max(0, x * 2),
        "halve": lambda x: max(0, x / 2),
        "int_double": lambda x: max(0, int(x * 2)),
        "int_halve": lambda x: max(0, int(x / 2))
    }

    def __init__(self, name, action_types, default):
        for action_type in action_types:
            if action_type not in self.actions.keys():
                raise Exception("Invalid action_type {}".format(action_type))
        self.name = name
        self.action_types = action_types
        self.default = default


class ListNode:
    def __init__(self, name, options, default):
        self.name = name
        self.options = options
        self.default = default


class SubScheme:
    def __init__(self, name, body):
        self.name = name
        self.body = body


class SchemeManager:
    __activation = OptionsNode("activation", [
        "sigmoid", "softmax", "elu", "selu", "softplus", "softsign", "relu",
        "tanh", "hard_sigmoid", "exponential", "linear"
    ])

    scheme = {
        "layers":
        ListNode(
            "layers", {
                "dense":
                ClassNode(
                    "dense", keras.layers.Dense, {
                        "activation":
                        __activation,
                        "units":
                        ValueNode("units", ["int_double", "int_halve"], 32)
                    })
            }, "dense"),
        "compile_args":
        SubScheme(
            "compile_args", {
                "optimizer":
                OptionsNode("optimizer", [
                    "nadam",
                    "adam",
                    "sgd",
                    "rmsprop",
                    "adagrad",
                    "adadelta",
                    "adamax",
                ]),
                "loss":
                OptionsNode("loss", [
                    "categorical_crossentropy", "mean_squared_error",
                    "mean_absolute_error", "mean_absolute_percentage_error",
                    "mean_squared_logarithmic_error", "squared_hinge", "hinge",
                    "categorical_hinge", "logcosh", "binary_crossentropy",
                    "kullback_leibler_divergence", "poisson",
                    "cosine_proximity"
                ])
            }),
        "patience":
        ValueNode("patience", ["increment", "decrement"], 1),
        "max_epochs":
        ValueNode("max_epochs", ["int_double", "int_halve"], 100)
    }

    def __get_class_map(self, a_scheme):
        class_map = {}
        for node in a_scheme.values():
            if isinstance(node, ClassNode):
                class_map[node.name] = node.class_ref
                class_map.update(self.__get_class_map(node.args))
            elif isinstance(node, ListNode):
                class_map.update(self.__get_class_map(node.options))
            elif isinstance(node, SubScheme):
                class_map.update(self.__get_class_map(node.body))
        return class_map

    def get_class_map(self):
        return self.__get_class_map(self.scheme)