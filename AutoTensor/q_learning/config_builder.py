from AutoTensor.q_learning.config_scheme import *


class ConfigBuilder:
    def __build_item(self, node):
        if isinstance(node, ValueNode):
            return node.default
        elif isinstance(node, OptionsNode):
            return node.options[node.default]
        elif isinstance(node, ClassNode):
            return {"class_name": node.name, "args": self.build(node.args)}
        elif isinstance(node, ListNode):
            return [self.__build_item(node.options[node.default])]

    def build(self, scheme):
        """
        Takes a scheme and using its defaults builds a config that
        can be edited by actions, used as states for the QLearner,
        and passed to the model_builder to build a tensorflow model.
        """
        config = {}
        for key, node in scheme.iteritems():
            if isinstance(node, SubScheme):
                config[key] = self.build(node.body)
            else:
                config[key] = self.__build_item(node)
        return config
