import numpy as np


class QLearner:
    def __init__(self,
                 discount=0.9,
                 alpha=0.2,
                 get_state=lambda state, action: "",
                 get_reward=lambda state: -1,
                 starting_state="",
                 actions=[]):
        # get_reward is a function that takes in a state
        # and returns that state's reward.
        self.get_reward = get_reward
        # get_state is a function that takes in a `state` and `action` and
        # returns the new state that results from taking `action` from that `state`.
        self.get_state = get_state
        self.discount = discount
        # `alpha` is like the learning rate of the QLearner.
        self.alpha = alpha
        # q_vals is a 2D matrix of q values, indexed by the state
        # as the row and the action as the column
        self.q_vals = np.zeros((1, len(actions)))
        self.actions = np.array(actions)
        # `states` maps the row indices in `q_vals` to their string
        # representations, which are more meaningful.
        reward_of_starting_state = self.get_reward(starting_state)
        self.states = np.array([starting_state])
        self.rewards = np.array([reward_of_starting_state])

    def __update_q(self, state_i):
        """
        Updates the q values for the current state and all its possible actions in the learner's q values table
        """
        for action_i, action in enumerate(self.actions):
            curr_q = self.q_vals[state_i, action_i]
            curr_state = self.states[state_i]
            new_state = self.get_state(curr_state, action)

            reward = self.get_reward(new_state)
            if new_state not in self.states:
                # Add this new found state to our records
                self.q_vals = np.append(self.q_vals,
                                        np.zeros((1, len(self.actions))))
                self.states = np.append(self.states, new_state)
            # Find the best future action to take from new state
            new_state_i = np.where(self.states == new_state)
            best_action_i = np.argmax(self.q_vals[new_state_i, :])

            # Finally, calculate the new Q-Value of this action state pair
            # and update it in the table
            new_q = curr_q + self.alpha * (
                (reward + self.discount *
                 self.q_vals[new_state_i, best_action_i]) - curr_q)
            self.q_vals[state_i, action_i] = new_q

    def find_state_with_best_q(self):
        """
        Iteratively calculate Q values, taking the optimal action at each state,
        until the q-value of the next state is no longer better than the last state.
        Warning: May cause the model to stop at a local maximum.
        """
        curr_best_q = -1
        curr_state_i = 0
        new_best_q = np.max(self.q_vals[curr_state_i, :])

        while new_best_q > curr_best_q:
            print("====New iteration of QLearner====")
            print("q_vals:\n{}".format(self.q_vals))
            print("states and rewards:\n{}".format(
                zip(self.states, self.rewards)))

            # Update the q values for the current state (explores all actions)
            self.__update_q(curr_state_i)
            # Find the next optimal state
            best_action = np.argmax(self.q_vals[curr_state_i])
            curr_state = self.states[curr_state_i]
            next_state = self.get_state(curr_state, best_action)
            # Update the loop values for the next iteration
            curr_best_q = new_best_q
            new_best_q = np.max(self.q_vals[curr_state_i])
            if new_best_q > curr_best_q:
                curr_state_i = np.where(self.states == next_state)

        print("========Finished========")
        print("Best Q-value: {}".format(curr_best_q))
        print("Best reward: {}".format(self.rewards[curr_state_i]))
        print("Best state:\n{}".format(self.states[curr_state_i]))
        return self.states[curr_state_i]
