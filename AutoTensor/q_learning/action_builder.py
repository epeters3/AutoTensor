from AutoTensor.q_learning.Action import Action
from AutoTensor.q_learning.config_scheme import ClassNode, OptionsNode, ValueNode, ListNode, SubScheme


class ActionBuilder:
    def make_name(self, elements):
        return "_".join(elements)

    def __add_value_actions(self, actions, action_path, node):
        for action_type in node.action_types:
            action_name = self.make_name([action_type, node.name])
            actions[action_name] = Action(action_path,
                                          ValueNode.actions[action_type])

    def __add_options_actions(self, actions, action_path, node):
        for option in node.options:
            action_name = self.make_name(["set", node.name, "to", option])
            actions[action_name] = Action(action_path, lambda _: option)

    def __add_list_actions(self, actions, action_path, node):
        pass

    def __add_class_actions(self, actions, action_path, node):
        pass
        # print("adding class actions for {}".format(node.name))
        # action_path.append(node.name)
        # action_path.append("args")
        # self.__build(node.args, actions, action_path)

    def build(self, scheme):
        actions = {}
        action_path = []
        self.__build(scheme, actions, action_path)
        return actions

    def __build(self, scheme, actions, action_path):
        """
        Traverses a config scheme and returns a dictionary of
        Action objects that can be used to alter instances of the scheme.
        """
        for key, node in scheme.iteritems():
            if isinstance(node, ValueNode):
                # This node represents a model parameter that is
                # a simple data structure (e.g. an int or float)
                self.__add_value_actions(actions, action_path + [key], node)
            elif isinstance(node, OptionsNode):
                # This node represents a model parameter that is
                # one of a list of options
                self.__add_options_actions(actions, action_path + [key], node)
            elif isinstance(node, ListNode):
                self.__add_list_actions(actions, action_path + [key], node)
            elif isinstance(node, ClassNode):
                # This node represents a model parameter that is
                # an instance of a class, and has arguments too
                self.__add_class_actions(actions, action_path + [key], node)
            elif isinstance(node, SubScheme):
                self.__build(node.body, actions, action_path + [key])
