from __future__ import division
import tensorflow as tf
from tensorflow import keras


class ClassNode:
    def __init__(self, class_ref, args):
        self.class_ref = class_ref
        self.args = args


class OptionsNode:
    def __init__(self, options):
        self.options = options
        self.default = 0


class ValueNode:
    actions = {
        "increment": lambda x: x + 1,
        "decrement": lambda x: max(0, x - 1),
        "increment5": lambda x: x + 5,
        "decrement5": lambda x: max(0, x - 5),
        "tuple_increment": lambda x: tuple([item + 1 for item in x]),
        "tuple_decrement": lambda x: tuple([max(0, item - 1) for item in x]),
        "double": lambda x: max(0, x * 2),
        "halve": lambda x: max(0, x / 2),
        "int_double": lambda x: max(0, int(x * 2)),
        "int_halve": lambda x: max(0, int(x / 2))
    }

    def __init__(self, action_types, default):
        for action_type in action_types:
            if action_type not in self.actions.keys():
                raise Exception("Invalid action_type {}".format(action_type))
        self.action_types = action_types
        self.default = default


class ListNode:
    def __init__(self, options, default):
        self.options = options
        self.default = default


class SubScheme:
    def __init__(self, body):
        self.body = body


class SchemeManager:
    __activation = OptionsNode([
        "sigmoid", "softmax", "elu", "selu", "softplus", "softsign", "relu",
        "tanh", "hard_sigmoid", "linear"
    ])

    scheme = {
        "layers":
        ListNode(
            {
                "dense":
                ClassNode(
                    keras.layers.Dense, {
                        "activation": __activation,
                        "units": ValueNode(["int_double", "int_halve"], 16)
                    }),
                # TODO: Add back in when adding new layers are supported actions
                # "conv2d":
                # ClassNode(
                #     keras.layers.Conv2D, {
                #         "filters": ValueNode(["int_double", "int_halve"], 16),
                #         "kernel_size": ValueNode(["increment", "decrement"], 3),
                #         "strides": ValueNode(["tuple_increment", "tuple_decrement"], (1,1)),
                #         "activation": __activation
                #     }),
            },
            "dense"),
        "compile_args":
        SubScheme({
            "optimizer":
            OptionsNode([
                "nadam",
                "adam",
                "sgd",
                "rmsprop",
                "adagrad",
                "adadelta",
                "adamax",
            ]),
            "loss":
            OptionsNode([
                "categorical_crossentropy", "mean_squared_error",
                "mean_absolute_error", "mean_absolute_percentage_error",
                "mean_squared_logarithmic_error", "squared_hinge", "hinge",
                "categorical_hinge", "logcosh", "binary_crossentropy",
                "kullback_leibler_divergence", "poisson", "cosine_proximity"
            ])
        }),
        "patience":
        ValueNode(["increment", "decrement"], 1),
        "max_epochs":
        ValueNode(["int_double", "int_halve"], 100)
    }

    def __get_class_map(self, a_scheme):
        class_map = {}
        for name, node in a_scheme.iteritems():
            if isinstance(node, ClassNode):
                print("{} is a ClassNode".format(name))
                class_map[name] = node.class_ref
                class_map.update(self.__get_class_map(node.args))
            elif isinstance(node, ListNode):
                class_map.update(self.__get_class_map(node.options))
            elif isinstance(node, SubScheme):
                class_map.update(self.__get_class_map(node.body))
        return class_map

    def get_class_map(self):
        """
        Builds a flat map of node names to class references,
        for all classes referenced in the scheme.
        """ 
        return self.__get_class_map(self.scheme)