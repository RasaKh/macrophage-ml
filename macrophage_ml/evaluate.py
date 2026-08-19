"""Nested donor-grouped evaluation. No operation learns from a held-out donor."""

import numpy as np


def group_folds(donors, count, seed):
    unique = sorted(set(donors))
    if len(unique) < count * 2:
        raise ValueError("Too few donors to form nontrivial grouped folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    folds = []
    for chosen in np.array_split(shuffled, count):
        test = np.flatnonzero(np.isin(donors, chosen))
        train = np.flatnonzero(~np.isin(donors, chosen))
        if set(donors[train]) & set(donors[test]):
            raise AssertionError("Donor leakage")
        folds.append((train, test))
    return folds
