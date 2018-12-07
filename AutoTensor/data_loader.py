import json

import arff
import numpy as np


def load_json_str(file_path):
    """Returns the json string from a .json file found at file_path"""
    with open(file_path) as f:
        data = json.load(f)
        return json.dumps(data)


def load_arff(file_path, ):
    with open(file_path) as f:
        dataset = arff.load(f, encode_nominal=True)
        return np.array(dataset["data"]), dataset


def split_last_col(data):
    data = data[:, :-1]
    labels = data[:, -1]
    return data, labels


def split_train_test(data, test_ratio, val_ratio):
    if test_ratio < 0 or test_ratio > 1 or val_ratio < 0 or val_ratio > 1:
        raise Exception("train_ratio must be between 0 and 1")
    if test_ratio + val_ratio > 1:
        raise Exception(
            "The sum of test_ratio and val_ratio cannot be greater than 1")

    np.random.shuffle(data)
    num_instances = data.shape[0]

    test_end = int(num_instances * test_ratio)
    val_end = test_end + int(num_instances * val_ratio)
    test = data[:test_end, :]
    val = data[test_end:val_end, :]
    train = data[val_end:, :]

    train_data, train_labels = split_last_col(train)
    test_data, test_labels = split_last_col(test)
    val_data, val_labels = split_last_col(val)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def get_arff_dataset(file_path, test_ratio, val_ratio):
    data, data_with_meta = load_arff(file_path)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_train_test(
        data, test_ratio, val_ratio)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels, data_with_meta


def get_my_arff():
    return get_arff_dataset("AutoTensor/data/iris.arff", 0.15, 0.15)