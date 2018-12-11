from __future__ import print_function
import json
from collections import namedtuple

import arff
import numpy as np

from AutoTensor.data_mgmt.data_shaper import normalize
# from AutoTensor.data.mushrooms.UnpackMushrooms import unpack as unpack_mushrooms

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
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels


def one_hot_encode(vector):
    classes = np.unique(vector)
    # Source: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    encoded = np.zeros((vector.shape[0], classes.size))
    encoded[np.arange(vector.shape[0]), vector.astype(int)] = 1
    return encoded


def split_dataset(data,
                  test_ratio,
                  val_ratio,
                  should_one_hot_encode,
                  labels=None):
    """
    Splits data and labels into training, validation, and test sets.
    If labels=None, it is assumed the last column of data contains the labels
    """
    if test_ratio < 0 or test_ratio > 1 or val_ratio < 0 or val_ratio > 1:
        raise Exception("train_ratio must be between 0 and 1")
    if test_ratio + val_ratio > 1:
        raise Exception(
            "The sum of test_ratio and val_ratio cannot be greater than 1")

    np.random.shuffle(data)
    num_instances = data.shape[0]

    test_end = int(num_instances * test_ratio)
    val_end = test_end + int(num_instances * val_ratio)
    if labels is None:
        data, labels = split_last_col(data)
    data = normalize(data)

    if should_one_hot_encode:
        labels = one_hot_encode(labels)

    test_data = data[:test_end, :]
    test_labels = labels[:test_end]

    val_data = data[test_end:val_end, :]
    val_labels = labels[test_end:val_end]

    train_data = data[val_end:, :]
    train_labels = labels[val_end:]
    print("train_data[0:2]={}\ntrain_labels[0:2]={}".format(
        train_data[0:2], train_labels[0:2]))

    return Data(train_data, train_labels, val_data, val_labels, test_data,
                test_labels)


def get_arff_dataset(file_path, test_ratio, val_ratio):
    data, data_with_meta = load_arff(file_path)
    data = split_dataset(data, test_ratio, val_ratio, True)
    num_classes = len(data_with_meta["attributes"][-1][1])
    print("arff dataset meta data:\n{}".format(data_with_meta["attributes"]))
    print("num_classes: {}".format(num_classes))
    return data, num_classes


# def get_my_mushrooms():
#     mushrooms = unpack_mushrooms()
#     image_arrays = np.array([mushroom.image_data for mushroom in mushrooms])
#     family_list = [
#         "Pluteaceae", "Boletaceae", "Cortinariaceae", "Russulaceae",
#         "Agaricaceae"
#     ]
#     labels = np.array(
#         [1 if mushroom.family in family_list else 0 for mushroom in mushrooms])
#     data = split_dataset(image_arrays, 0.15, 0.15, True, labels)
#     num_classes = data.train_labels.shape[1]
#     return data, num_classes


def get_my_arff_dataset():
    return get_arff_dataset("AutoTensor/data/iris.arff", 0.15, 0.15)
