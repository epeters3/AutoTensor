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

    def do(self, state):
        """
        :type state: {}
        :return: {}
        """
        print("now doing at path: {}".format(str(self.path)))
        state_copy = copy.deepcopy(state)
        sub_data = state_copy
        for step in self.path[:-1]:
            # descend through state, stopping one short
            try:
                sub_data = sub_data[step]
            except:
                print("Failed accessing step {} of sub_data {}".format(
                    step, sub_data))
                raise
        # Perform the action on the data at the last step in self.path
        sub_data[self.path[-1]] = self.perform(sub_data[self.path[-1]])
        # Return the copy of state, with action having been performed on it.
        return state_copy
