from typing import NamedTuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Dataset(NamedTuple):
    train_data: pd.DataFrame
    train_labels: pd.Series
    val_data: pd.DataFrame
    val_labels: pd.Series
    test_data: pd.DataFrame
    test_labels: pd.Series
    num_classes: int


def prepare_dataset(
    X: pd.DataFrame, y: pd.Series, val_ratio: float, test_ratio: float
) -> Dataset:
    """
    Splits data and labels into training, validation, and test sets.
    """
    if test_ratio < 0 or test_ratio > 1 or val_ratio < 0 or val_ratio > 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not test_ratio + val_ratio < 1:
        raise ValueError("the sum of test_ratio and val_ratio must be less than 1")

    num_instances = X.shape[0]
    num_classes = y.nunique()

    X, y = shuffle(X, y)

    normalizer = StandardScaler()
    X = normalizer.fit_transform(X)

    # TODO: Encode all categorical labels in X, or do
    # a one-hot-encoding

    # One hot encode y
    encoder = LabelEncoder()
    binarizer = LabelBinarizer()
    y = encoder.fit_transform(y)
    y = binarizer.fit_transform(y)

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

    print(f"train_data.shape: {train_data.shape}")
    print(f"val_data.shape: {val_data.shape}")
    print(f"test_data.shape: {test_data.shape}")

    return Dataset(
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        num_classes,
    )
