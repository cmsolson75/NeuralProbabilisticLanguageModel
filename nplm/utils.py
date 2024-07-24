import torch


class CharUtils:
    def __init__(self):
        self.unique_chars = [
            ".",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]
        self._string_to_int = {s: i for i, s in enumerate(self.unique_chars)}
        self._int_to_string = {i: s for s, i in self._string_to_int.items()}

    def stoi(self, s: str) -> int:
        """Convert a string character to its integer index."""
        return self._string_to_int[s]

    def itos(self, i: int) -> str:
        """Convert an integer index to its string character."""
        return self._int_to_string[i]


def get_device() -> str:
    """Return the appropriate device type ('cuda' or 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"
