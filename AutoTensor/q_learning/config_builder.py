from AutoTensor.q_learning.config_scheme import *


class ConfigBuilder:
    def __build_item(self, node, name):
        if isinstance(node, ValueNode):
            return node.default
        elif isinstance(node, OptionsNode):
            return node.options[node.default]
        elif isinstance(node, ClassNode):
            return {"class_name": name, "args": self.build(node.args)}
        elif isinstance(node, ListNode):
            return [
                self.__build_item(node.options[node.default], node.default)
            ]

    def build(self, scheme):
        """
        Takes a scheme and using its defaults builds a config that
        can be edited by actions, used as states for the QLearner,
        and passed to the model_builder to build a tensorflow model.
        """
        config = {}
        for name, node in scheme.items():
            if isinstance(node, SubScheme):
                config[name] = self.build(node.body)
            else:
                config[name] = self.__build_item(node, name)
        return config
