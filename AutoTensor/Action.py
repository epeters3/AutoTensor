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
        state_copy = copy.deepcopy(state)
        sub_data = state_copy
        for step in self.path[:-1]:
            # descend through state, stopping one short
            sub_data = sub_data[step]
        # Perform the action on the data at the last step in self.path
        sub_data[self.path[-1]] = self.perform(sub_data[self.path[-1]])
        # Return the copy of state, with action having been performed on it.
        return state_copy
