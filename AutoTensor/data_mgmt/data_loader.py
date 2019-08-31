import json
from collections import namedtuple
import re
from typing import NamedTuple

from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from AutoTensor.data_mgmt.data_shaper import normalize


class DataSet(NamedTuple):
    train_data: pd.DataFrame
    train_labels: pd.Series
    val_data: pd.DataFrame
    val_labels: pd.Series
    test_data: pd.DataFrame
    test_labels: pd.Series


def prepare_dataset(
    X: pd.DataFrame, y: pd.Series, test_ratio: float, val_ratio: float
) -> DataSet:
    """
    Splits data and labels into training, validation, and test sets.
    """
    if test_ratio < 0 or test_ratio > 1 or val_ratio < 0 or val_ratio > 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not test_ratio + val_ratio < 1:
        raise ValueError("the sum of test_ratio and val_ratio must be less than 1")

    num_instances = X.shape[0]
    X, y = shuffle(X, y)

    normalizer = StandardScaler()
    X = normalizer.fit_transform(X)

    # One hot encode y
    encoder = LabelEncoder()
    binarizer = LabelBinarizer()
    y = encoder.fit_transform(y)
    y = binarizer.fit_transform(y)
    print(f"one hot encoded y: {y}")

    rest_ratio = val_ratio + test_ratio
    train_data, rest_data, train_labels, rest_labels = train_test_split(
        X, y, train_size=1 - rest_ratio, test_size=rest_ratio
    )

    val_data, test_data, val_labels, test_labels = train_test_split(
        rest_data,
        rest_labels,
        train_size=val_ratio / rest_ratio,
        test_size=test_ratio / rest_ratio,
    )

    print(
        "train_data[0:2]={}\ntrain_labels[0:2]={}".format(
            train_data[0:2], train_labels[0:2]
        )
    )

    return DataSet(
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    )


def get_arff_dataset(file_path, target_index, test_ratio, val_ratio):
    data_arr, _ = arff.loadarff(file_path)
    data_arr = pd.DataFrame(data_arr)
    x_indices = [i for i in range(data_arr.shape[1]) if i != target_index]
    target_col_name = data_arr.columns[target_index]
    X = data_arr.drop(columns=target_col_name)
    y = pd.Series(data_arr[target_col_name], dtype="category")
    data = prepare_dataset(X, y, test_ratio, val_ratio)
    num_classes = y.nunique()
    return data, num_classes


def get_file_ext(file_path):
    file_ext_regex = r"\.[^\.]+$"
    return re.search(file_ext_regex, file_path)[0]


file_ext_to_loader = {".arff": get_arff_dataset}


def load_dataset_file(path, target_index):
    ext = get_file_ext(path)
    # TODO: add test_ratio and val_ratio as CLI args
    return file_ext_to_loader[ext](path, target_index, 0.15, 0.15)
