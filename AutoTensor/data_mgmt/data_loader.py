import re
from typing import Tuple, Callable, Dict

from scipy.io import arff
import pandas as pd


def split_into_x_and_y(
    data: pd.DataFrame, target_index: int
) -> Tuple[pd.DataFrame, pd.Series]:
    target_col_name = data.columns[target_index]
    X = data.drop(columns=target_col_name)
    # We assume y is categorical, and this is a classification problem.
    # TODO: support regression.
    y = pd.Series(data[target_col_name], dtype="category")
    return X, y


def load_arff(file_path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    data, _ = arff.loadarff(file_path)
    data = pd.DataFrame(data)
    return split_into_x_and_y(data, target_index)


def load_csv(file_path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(file_path)
    return split_into_x_and_y(data, target_index)


def get_file_ext(file_path: str) -> str:
    file_ext_regex = r"\.[^\.]+$"
    return re.search(file_ext_regex, file_path)[0]


file_ext_to_loader: Dict[str, Callable[[str, int], Tuple[pd.DataFrame, pd.Series]]] = {
    ".arff": load_arff,
    ".csv": load_csv,
}


def load_dataset_file(path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    ext = get_file_ext(path)
    assert (
        ext in file_ext_to_loader.keys()
    ), f"We currently do not supporting loading {ext} files. We'd love for you to make a PR!"
    return file_ext_to_loader[ext](path, target_index)
