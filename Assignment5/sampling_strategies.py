# sampling_strategies.py
# Simple functions for choosing the next character during text generation.

import numpy as np


def greedy_sampling(probabilities):
    """Choose the character with the highest probability."""
    return int(np.argmax(probabilities))


def temperature_sampling(probabilities, temperature=1.0):
    """Choose a character using temperature."""
    probabilities = np.asarray(probabilities, dtype=float)

    # Avoid dividing by zero
    temperature = max(temperature, 0.01)

    # Change the probabilities using temperature
    probabilities = np.log(probabilities + 1e-8) / temperature
    probabilities = np.exp(probabilities - np.max(probabilities))
    probabilities = probabilities / probabilities.sum()

    return int(np.random.choice(len(probabilities), p=probabilities))


def top_k_sampling(probabilities, k=5):
    """Choose a character from the k most likely characters."""
    probabilities = np.asarray(probabilities, dtype=float)

    # Get indexes of the k largest probabilities
    top_indexes = np.argsort(probabilities)[-k:]

    # Keep only these probabilities
    top_probabilities = probabilities[top_indexes]
    top_probabilities = top_probabilities / top_probabilities.sum()

    return int(np.random.choice(top_indexes, p=top_probabilities))


def top_p_sampling(probabilities, p=0.9):
    """Choose a character from the smallest group reaching probability p."""
    probabilities = np.asarray(probabilities, dtype=float)

    # Sort probabilities from largest to smallest
    sorted_indexes = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indexes]

    # Find the group whose total probability reaches p
    cumulative_probabilities = np.cumsum(sorted_probabilities)
    keep = cumulative_probabilities - sorted_probabilities < p
    top_indexes = sorted_indexes[keep]

    # Randomly choose from the selected group
    top_probabilities = probabilities[top_indexes]
    top_probabilities = top_probabilities / top_probabilities.sum()

    return int(np.random.choice(top_indexes, p=top_probabilities))


def choose_next_character(probabilities, strategy="temperature",
                          temperature=1.0, k=5, p=0.9):
    """Choose a sampling method by its name."""

    if strategy == "greedy":
        return greedy_sampling(probabilities)

    elif strategy == "temperature":
        return temperature_sampling(probabilities, temperature)

    elif strategy == "top_k":
        return top_k_sampling(probabilities, k)

    elif strategy == "top_p":
        return top_p_sampling(probabilities, p)

    else:
        raise ValueError("Please choose greedy, temperature, top_k or top_p.")
