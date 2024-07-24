from nplm.model import MLP
from nplm.utils import CharUtils

import torch
import torch.nn.functional as F
import argparse
import json

from typing import List, Tuple

char_utils = CharUtils()


def clean_input(prompt: str) -> str:
    """Clean input prompt by only keeping alphabetic characters, and convert to lowercase."""
    unique_chars = [
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
    return "".join((x for x in prompt if x in unique_chars)).strip().lower()


def truncate_prompt(
    tokenized_prompt: List[int], max_length: int
) -> Tuple[List[int], List[int]]:
    """Truncate the tokenized prompt to fit within the maximum context size."""
    if len(tokenized_prompt) > max_length:
        return tokenized_prompt[-max_length:], tokenized_prompt[:-max_length]
    return tokenized_prompt, []


def initialize_context(prompt: str, context_size: int) -> Tuple[List[int], List[int]]:
    """Initialize the context window and token list from the given prompt."""
    clean_prompt = clean_input(prompt)
    tokenized_prompt = [char_utils.stoi(char) for char in clean_prompt]
    truncated_prompt, remainder = truncate_prompt(tokenized_prompt, context_size)
    context_window = [0] * (context_size - len(truncated_prompt)) + truncated_prompt
    return context_window, remainder + truncated_prompt


def generate_name(
    model: MLP, context_size: int, prompt: str, temperature: float
) -> str:
    """Generate a single name based on the prompt using the model"""
    context_window, tokenized_output = initialize_context(prompt, context_size)
    if temperature <= 0:
        raise ValueError("Temperature must be above zero")

    while True:
        logits = model(context_window)
        logits = logits / temperature
        probabilities = F.softmax(logits, dim=1)
        next_char_index = torch.multinomial(probabilities, num_samples=1).item()
        context_window = context_window[1:] + [next_char_index]
        tokenized_output.append(next_char_index)
        if next_char_index == 0:  # End of name token
            break
    return (
        "".join(char_utils.itos(index) for index in tokenized_output)
        .strip(".")
        .capitalize()
    )


def run_app() -> None:
    """Run the name generation app with the specified prompt and number of names."""
    with open("configs/app_config.json") as c:
        config = json.load(c)

    model = MLP.init_model_from_load(config["modelPath"])
    model.eval()
    context_size = model.model_dims["context"]

    parser = argparse.ArgumentParser(description="Generate Names")
    parser.add_argument(
        "--prompt", type=str, help="Prompt to generate name with", default=""
    )
    parser.add_argument(
        "--num_names", type=int, help="Number of names to generate", default=1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for name generation",
        default=config["defaultTemperature"],
    )
    args = parser.parse_args()

    for _ in range(args.num_names):
        generated_name = generate_name(
            model, context_size, args.prompt, args.temperature
        )
        print(generated_name)


if __name__ == "__main__":
    run_app()
