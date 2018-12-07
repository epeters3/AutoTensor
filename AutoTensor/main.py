from AutoTensor.QLearner import QLearner
from AutoTensor.tensor_q_mapper import actions, starting_state, compose_get_state, compose_get_reward
from AutoTensor.data_loader import get_my_arff, load_json_str
import pprint

state = load_json_str("AutoTensor/data/nn_initial_state.json")


def find_optimal_model(get_dataset):
    train_data, train_labels, val_data, val_labels, test_data, test_labels, meta = get_dataset(
    )
    pprint.pprint("dataset:\n{}".format(meta))

    get_state = compose_get_state()
    get_reward = compose_get_reward(train_data, train_labels, val_data,
                                    val_labels, test_data, test_labels)
    acc = get_reward(state)
    print(
        "======== FINISHED WITH TEST OF CURRYING AND TENSOR FLOW - ACCURACY IS {} ========"
        .format(acc))

    # qlearner = QLearner(
    #     discount=0.9,
    #     alpha=0.2,
    #     get_state=get_state,
    #     get_reward=get_reward,
    #     starting_state=starting_state,
    #     actions=actions)

    # qlearner.find_state_with_best_q()


if __name__ == "__main__":
    find_optimal_model(get_my_arff)