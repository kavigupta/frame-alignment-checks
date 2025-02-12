from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from permacache import stable_hash
from torch import nn


def all_seqs(n, *, amount=4):
    """
    Generates all sequences of length n, with the given amount of nucleotides.
    """
    if n == 0:
        yield []
        return
    for s in all_seqs(n - 1, amount=amount):
        for i in np.eye(amount):
            yield [i] + s


@lru_cache(None)
def all_3mers() -> np.ndarray:
    """
    The 64 possible 3mers. We always use this order for consistency.

    :returns: The 64 possible 3mers. Will be of shape (64, 4).
    """
    return np.array(list(all_seqs(3)))


def stable_hash_cached(model):
    """
    Cache the stable hash of a model.

    Note: **only use this function if the model is not going to change**.
    """
    # pylint: disable=protected-access
    if model is None:
        return stable_hash(None)
    assert not model.training
    if not hasattr(model, "_stable_hash_value"):
        model._stable_hash_value = stable_hash(model)
    return model._stable_hash_value


def collect_windows(x, locs, cl):
    """
    Collect windows around the given locations in the sequence. Extra padding of zeros will be added
    if the windows go out of bounds.

    :param x: The sequence to collect windows from.
    :param locs: The locations to collect windows around.
    :param cl: The context length of the windows.

    :returns:
        x_windows: The windows around the locations. Will be of shape (len(locs), cl + 1, 4).
    """
    assert cl % 2 == 0, "cl must be even"
    x_window = np.zeros((len(locs), cl + 1, 4), dtype=np.float32)
    for i, loc in enumerate(locs):
        start_in_x, end_in_x = loc - cl // 2, loc + cl // 2 + 1
        start_in_x_window, end_in_x_window = 0, cl + 1
        if start_in_x < 0:
            start_in_x_window += -start_in_x
            start_in_x = 0
        if end_in_x >= len(x):
            end_in_x_window -= end_in_x - len(x)
            end_in_x = len(x)
        x_window[i, start_in_x_window:end_in_x_window] = x[start_in_x:end_in_x]
    return x_window


def extract_center(model, xs):
    """
    Run the model on the given sequences and extract the center of the output.

    :param model: The model to run.
    :param xs: The sequences to run the model on. Should be of shape (N, cl + 1, 4).

    :returns:
        yps: The center of the output of the model. Will be of shape (N, 3).
    """
    yps = model(xs)
    yps = yps[:, yps.shape[1] // 2]
    yps = yps.softmax(-1)
    return yps


def bootstrap_series(ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a 95% confidence interval for the mean of the given series at each point, using a bootstrap.
    Boostraps are done at each time point independently.

    This is done deterministically with a fixed random seed.

    :param ys: The series to compute the confidence interval for. Shape: (N, T) where T is the number of
        time points and N is the number of samples at each point.
    :returns:
        lo: The lower bound of the confidence interval at each time point. Shape: (T,).
        hi: The upper bound of the confidence interval at each time point. Shape: (T,).
    """
    bootstrap = ys[np.random.RandomState(0).choice(len(ys), size=(len(ys), 10_000))]
    bootstrap = bootstrap.mean(0)
    lo, hi = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return lo, hi


def draw_bases(xs: np.ndarray):
    """
    Draw the bases from the given array.

    :param xs: The array to draw the bases from. Should be of shape (..., 4) and one-hot encoded, or
        of shape (...,) and integer encoded with all values in (0, 1, 2, 3). Will be returned as a
        nested list of strings, where (N, 4) will be returned as a string of length N and (N, M, 4)
        will be returned as a list of N strings of length M.
    """
    if np.issubdtype(xs.dtype, np.integer) and 0 < xs.max() < 4:
        xs = np.eye(4)[xs]
    assert xs.shape[-1] == 4 and len(xs.shape) > 1
    if len(xs.shape) == 2:
        mask = (xs == 0).all(-1)
        xs = xs.argmax(-1)
        xs = np.array(list("ACGT"))[xs]
        xs[mask] = "N"
        return "".join(xs)
    return [draw_bases(x) for x in xs]


def display_permutation_test_p_values(results: Dict[Tuple[str, str], float], title):
    """
    Print the p values of the permutation test for the given results. The results should be a dictionary
    where the keys are the names of the models and the values are the results for each seed.
    """
    print(f"P value of comparison: {title}")
    for (k1, k2), p in results.items():
        print(f"{k1:20s} {k2:20s} {p:.4f} {'*' if p < 0.05 else ''}")


def permutation_test(xs, ys, count=10**4):
    """
    Perform a permutation test to determine the p value of the difference of means between xs and ys,
    where the null hypothesis is that the means are the same. Returns a two-tailed p value.
    """
    complete = np.concatenate([xs, ys])
    rng = np.random.RandomState(0)
    results = np.zeros((complete.shape[0], count - 1))
    for i in range(results.shape[1]):
        rng.shuffle(complete)
        results[:, i] = complete[:]
    xs_permute, ys_permute = results[: len(xs)], results[len(xs) :]
    bad = (
        np.abs(np.mean(xs) - np.mean(ys))
        <= np.abs(xs_permute.mean(0) - ys_permute.mean(0))
    ).sum()
    return (bad + 1) / (results.shape[1] + 1)


def parse_sequence_as_one_hot(nt_sequence: str) -> np.ndarray:
    """
    Parse a string genomic sequence as one hot encoding. The sequence should only contain A, C, G, T, and N.

    :param nt_sequence: The nucleotide sequence to parse.
    :returns: The one hot encoding of the sequence, with shape (len(nt_sequence), 4). One hot encoding is
        in the order A, C, G, T, with N being all zeros.
    """
    table = np.concatenate(
        [np.eye(4, dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8)]
    )
    return table[["ACGTN".index(nt) for nt in nt_sequence]]


def device_of(m: nn.Module) -> torch.device:
    """
    Get the device of a module.
    """
    return next(m.parameters()).device
