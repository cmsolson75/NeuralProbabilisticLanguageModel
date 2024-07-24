import random
import torch
from typing import List, Tuple
from .utils import CharUtils


def load_data(data_path: str) -> List[str]:
    """Loads data from the specified data path."""
    with open(data_path) as f:
        return f.read().splitlines()


def create_dataset(
    words: List[str], context_size: int, char_utils: CharUtils
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates a dataset from words with the given context size and character utilities."""
    xs, ys = [], []

    for word in words:
        context = [0] * context_size
        for char in word + ".":
            ix = char_utils.stoi(char)
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]

    xs_tensor = torch.tensor(xs, dtype=torch.long)
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return xs_tensor, ys_tensor


def create_random_split(
    data: List[str], split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[int, int]:
    """Splits the data into train, dev, and test sets based on the given split ratios."""
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_p, dev_p, _ = split
    n = len(data)
    split_point_one = int(n * train_p)
    split_point_two = int(n * (train_p + dev_p))
    return split_point_one, split_point_two


def load_datasets(context_size: int, data_path: str, shuffle: bool = True) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """Loads the datasets, shuffling them if required, and returns train, dev, and test sets."""
    name_data = load_data(data_path)
    if shuffle:
        random.shuffle(name_data)

    char_utils = CharUtils()
    n1, n2 = create_random_split(name_data)

    train_data = create_dataset(
        name_data[:n1], context_size=context_size, char_utils=char_utils
    )
    dev_data = create_dataset(
        name_data[n1:n2], context_size=context_size, char_utils=char_utils
    )
    test_data = create_dataset(
        name_data[n2:], context_size=context_size, char_utils=char_utils
    )

    return train_data, dev_data, test_data
