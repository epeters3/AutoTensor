import copy


class Action():
    """Represents a thing that performs an action at some path in a dictionary tree"""

    def __init__(self, path, perform):
        """
        :type path: []
        :type perform: func
        """
        self.path = path
        self.perform = perform

    def descend(self, data, key):
        return data[key]

    def __do(self, state, path):
        """
        :type state: {}
        :return: {}
        """
        print("now doing at path: {}".format(str(path)))
        sub_data = state
        steps_done = 0
        for step in path[:-1]:
            # descend through state, stopping one short
            try:
                if isinstance(sub_data, dict):
                    sub_data = self.descend(sub_data, step)
                elif isinstance(sub_data, list):
                    for node in sub_data:
                        if node["class_name"] == step:
                            sub_data = self.__do(node, path[steps_done + 1:])
                else:
                    raise Exception("Unsupported type found in Action.do")
            except:
                print("Failed accessing sub_data {} with step {} in path {}".
                      format(sub_data, step, path))
                raise
            steps_done += 1
        # Perform the action on the data at the last step in path
        sub_data[path[-1]] = self.perform(sub_data[path[-1]])
        # Return the copy of state, with action having been performed on it.
        return state

    def do(self, state):
        state_copy = copy.deepcopy(state)
        return self.__do(state_copy, self.path)
