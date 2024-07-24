import torch
import math
from typing import Tuple, Union


def xavier_normal(
    *size: Union[int, Tuple[int, ...]], generator: torch.Generator = None
) -> torch.Tensor:
    """Initializes a tensor with Xavier normal distribution.

    Args:
        size: The dimensions of the tensor.
        generator: A random number generator.

    Returns:
        A tensor initialized with Xavier normal distribution.
    """
    std = math.sqrt(2 / sum(size))
    return torch.normal(0, std, size, generator=generator).type(torch.float32)


def xavier_uniform(
    *size: Union[int, Tuple[int, ...]], generator: torch.Generator = None
) -> torch.Tensor:
    """Initializes a tensor with Xavier uniform distribution.

    Args:
        size: The dimensions of the tensor.
        generator: A random number generator.

    Returns:
        A tensor initialized with Xavier uniform distribution.
    """
    t = torch.empty(size)
    scaling_factor = math.sqrt(6 / sum(size))
    return t.uniform_(-scaling_factor, scaling_factor, generator=generator).type(
        torch.float32
    )


# normal -> mean = 0, std = 1 / number_inputs
# 1000 -> sum(input * weights(std 1))