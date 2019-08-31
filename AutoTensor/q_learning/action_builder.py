from AutoTensor.q_learning.Action import Action
from AutoTensor.q_learning.config_scheme import ClassNode, OptionsNode, ValueNode, ListNode, SubScheme


class ActionBuilder:
    def make_name(self, elements):
        return "_".join(elements)

    def __add_value_actions(self, actions, action_path, node, name):
        for action_type in node.action_types:
            action_name = self.make_name([action_type, name])
            actions[action_name] = Action(action_path,
                                          ValueNode.actions[action_type])

    def __add_options_actions(self, actions, action_path, node, name):
        for option in node.options:
            action_name = self.make_name(["set", name, "to", option])
            actions[action_name] = Action(
                action_path, lambda _, opt=option: opt)

    def __add_class_actions(self, actions, action_path, node, name):
        new_path = action_path + [name, "args"]
        self.__build(node.args, actions, new_path, True)

    def build(self, scheme):
        actions = {}
        action_path = []
        self.__build(scheme, actions, action_path, True)
        return actions

    def __build(self, scheme, actions, action_path, include_path):
        """
        Traverses a config scheme and returns a dictionary of
        Action objects that can be used to alter instances of the scheme.
        """
        for name, node in scheme.items():
            new_path = action_path + [name] if include_path else action_path
            if isinstance(node, ValueNode):
                # This node represents a model parameter that is
                # a simple data structure (e.g. an int or float)
                self.__add_value_actions(actions, new_path, node, name)
            elif isinstance(node, OptionsNode):
                # This node represents a model parameter that is
                # one of a list of options
                self.__add_options_actions(actions, new_path, node, name)
            elif isinstance(node, ListNode):
                self.__build(node.options, actions, new_path, False)
            elif isinstance(node, ClassNode):
                # This node represents a model parameter that is
                # an instance of a class, and has arguments too
                self.__add_class_actions(actions, new_path, node, name)
            elif isinstance(node, SubScheme):
                self.__build(node.body, actions, new_path, True)
