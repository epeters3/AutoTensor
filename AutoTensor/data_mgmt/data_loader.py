import re
from typing import Tuple, Callable, Dict

from scipy.io import arff
import pandas as pd


def load_arff(file_path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    data_arr, _ = arff.loadarff(file_path)
    data_arr = pd.DataFrame(data_arr)
    target_col_name = data_arr.columns[target_index]
    X = data_arr.drop(columns=target_col_name)
    y = pd.Series(data_arr[target_col_name], dtype="category")
    return X, y


def load_csv(file_path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    data_arr = pd.read_csv(file_path)
    target_col_name = data_arr.columns[target_index]
    X = data_arr.drop(columns=target_col_name)
    y = pd.Series(data_arr[target_col_name], dtype="category")
    return X, y


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
