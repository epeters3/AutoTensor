import json
from collections import namedtuple

import arff
import numpy as np

from AutoTensor.data_mgmt.data_shaper import normalize

Data = namedtuple("Data", [
    "train_data", "train_labels", "val_data", "val_labels", "test_data",
    "test_labels"
])


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


def split_dataset(data, test_ratio, val_ratio):
    if test_ratio < 0 or test_ratio > 1 or val_ratio < 0 or val_ratio > 1:
        raise Exception("train_ratio must be between 0 and 1")
    if test_ratio + val_ratio > 1:
        raise Exception(
            "The sum of test_ratio and val_ratio cannot be greater than 1")

    np.random.shuffle(data)
    num_instances = data.shape[0]

    test_end = int(num_instances * test_ratio)
    val_end = test_end + int(num_instances * val_ratio)
    data, labels = split_last_col(data)
    data = normalize(data)

    test_data = data[:test_end, :]
    test_labels = labels[:test_end]

    val_data = data[test_end:val_end, :]
    val_labels = labels[test_end:val_end]

    train_data = data[val_end:, :]
    train_labels = labels[val_end:]

    return Data(train_data, train_labels, val_data, val_labels, test_data,
                test_labels)


def get_arff_dataset(file_path, test_ratio, val_ratio):
    data, data_with_meta = load_arff(file_path)
    data = split_dataset(data, test_ratio, val_ratio)
    num_classes = len(data_with_meta["attributes"][-1][1])
    print("arff dataset meta data:\n{}".format(data_with_meta["attributes"]))
    print("num_classes: {}".format(num_classes))
    return data, num_classes


def get_my_arff():
    return get_arff_dataset("AutoTensor/data/iris.arff", 0.15, 0.15)