import re
from typing import Tuple

from scipy.io import arff
import pandas as pd


def load_arff(file_path, target_index) -> Tuple[pd.DataFrame, pd.Series]:
    data_arr, _ = arff.loadarff(file_path)
    data_arr = pd.DataFrame(data_arr)
    target_col_name = data_arr.columns[target_index]
    X = data_arr.drop(columns=target_col_name)
    y = pd.Series(data_arr[target_col_name], dtype="category")
    return X, y


def get_file_ext(file_path: str) -> str:
    file_ext_regex = r"\.[^\.]+$"
    return re.search(file_ext_regex, file_path)[0]


file_ext_to_loader = {".arff": load_arff}


def load_dataset_file(path: str, target_index: int) -> Tuple[pd.DataFrame, pd.Series]:
    ext = get_file_ext(path)
    return file_ext_to_loader[ext](path, target_index)
