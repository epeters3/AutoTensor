import numpy as np
import json

from AutoTensor.utils import print_no_nl


class QLearner:
    def __init__(
        self,
        discount=0.9,
        alpha=0.2,
        get_state=lambda state, action: "",
        get_reward=lambda state: -1,
        starting_state="",
        actions=[],
        report_file_path="",
    ):
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
        self.report_file_path = report_file_path

    def __print_q_vals(self):
        to_print = np.append(
            np.reshape(self.actions, (1, len(self.actions))), self.q_vals, axis=0
        )
        print("==== Q Values ====")
        print(to_print)

    def __update_q(self, state_i):
        """
        Updates the q values for the current state and all its possible actions in the learner's q values table
        """
        curr_state = self.states[state_i]
        best_reward, best_action, best_state = -1, "no_action_selected", {}
        for action_i, action in enumerate(self.actions):
            curr_q = self.q_vals[state_i, action_i]
            new_state = self.get_state(curr_state, action)
            print_no_nl(f"{action} ({action_i+1}/{len(self.actions)}) ==> ")
            reward = self.get_reward(new_state)
            print(" ==> {:.4f}".format(reward))
            if new_state not in self.states:
                # Add this new found state to our records
                self.q_vals = np.append(
                    self.q_vals, np.zeros((1, len(self.actions))), axis=0
                )

                self.states = np.append(self.states, new_state)
                self.rewards = np.append(self.rewards, reward)
            # Find the best future action to take from new state
            new_state_i = np.where(self.states == new_state)
            best_action_i = np.argmax(self.q_vals[new_state_i, :])

            # Finally, calculate the new Q-Value of this action state pair
            # and update it in the table
            new_q = curr_q + self.alpha * (
                (reward + self.discount * self.q_vals[new_state_i, best_action_i])
                - curr_q
            )
            self.q_vals[state_i, action_i] = new_q
            if reward > best_reward:
                best_reward = reward
                best_action = action
                best_state = new_state
        return best_reward, best_action, best_state

    def find_state_with_best_q(self):
        """
        Iteratively calculate Q values, taking the optimal action at each state,
        until the q-value of the next state is no longer better than the last state.
        Warning: May cause the model to stop at a local maximum.
        """
        curr_best_q = -1
        curr_state_i = 0
        new_best_q = np.max(self.q_vals[curr_state_i, :])
        optimal_path = []
        while new_best_q > curr_best_q:
            print("======== New iteration of QLearner ========")
            # self.__print_q_vals()
            print("Optimal path so far (state, action, reward):")
            print(optimal_path)

            # Update the q values for the current state (explores all actions)
            best_reward, best_action, next_state = self.__update_q(curr_state_i)
            # Find the next optimal state
            curr_state = self.states[curr_state_i]
            optimal_path.append((curr_state, best_action, best_reward))
            # Update the loop values for the next iteration
            curr_best_q = new_best_q
            new_best_q = np.max(self.q_vals[curr_state_i])
            if new_best_q > curr_best_q:
                curr_state_i = np.asscalar(np.argwhere(self.states == next_state)[0])

        print("========Finished========")
        print(f"Best Q-value: {curr_best_q}")
        print(f"Best reward: {self.rewards[curr_state_i]}")
        print(f"Best state:\n{self.states[curr_state_i]}")
        with open(self.report_file_path, "w") as f:
            results = [
                {"model_config": config, "test_acc": reward}
                for config, reward in zip(self.states, self.rewards)
            ]
            json.dump(results, f)
        return self.states[curr_state_i]
