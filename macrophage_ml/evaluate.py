"""Nested donor-grouped evaluation. No operation learns from a held-out donor."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CONDITIONS, PANEL


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
            # bug here? this assert is the only thing keeping donors apart
            # across the split -- it fired once before I seeded the rng, leave
            # it in for now
            raise AssertionError("Donor leakage")
        folds.append((train, test))
    return folds


class TopVariance(BaseEstimator, TransformerMixin):
    def __init__(self, k=1500):
        self.k = k

    def fit(self, X, y=None):
        variances = np.nanvar(X, axis=0)
        self.indices_ = np.sort(np.argsort(-variances, kind="stable")[:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return X[:, self.indices_]


def candidates(procedure):
    if procedure == "dummy": return [{}]
    if procedure == "thesis_lr": return [{"C": c} for c in (.01, .1, 1, 10)]
    if procedure.startswith("compact_lr"):
        sizes = (int(procedure.split("k")[-1]),) if "_k" in procedure else (5, 10, 20, 50)
        return [{"k": k, "C": c} for k in sizes for c in (.01, .1, 1, 10)]
    if procedure == "broad_lr":
        return [{"C": c, "l1_ratio": ratio} for c in (.01, .1, 1, 10) for ratio in (.1, .5, .9)]
    if procedure == "compact_gb":
        return [{"k": k, "depth": d, "learning_rate": rate} for k in (5, 10, 20, 50)
                for d in (1, 2) for rate in (.03, .1)]
    raise ValueError(procedure)


def build(procedure, params):
    if procedure == "dummy": return DummyClassifier(strategy="prior")
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if procedure.startswith("compact"):
        steps.append(("select", SelectKBest(f_classif, k=params["k"])))
    elif procedure == "broad_lr":
        steps.append(("select", TopVariance(k=1500)))
    if procedure == "compact_gb":
        from sklearn.ensemble import GradientBoostingClassifier
        steps.append(("model", GradientBoostingClassifier(n_estimators=100, max_depth=params["depth"],
                    learning_rate=params["learning_rate"], min_samples_leaf=3, random_state=42)))
    else:
        steps.append(("scale", StandardScaler()))
        if procedure == "broad_lr":
            model = LogisticRegression(C=params["C"], penalty="elasticnet", l1_ratio=params["l1_ratio"],
                                       solver="saga", max_iter=1200, tol=.003, random_state=42)
        else:
            model = LogisticRegression(C=params["C"], solver="lbfgs", max_iter=1000)
        steps.append(("model", model))
    return Pipeline(steps)


def matrix_for(procedure, all_genes, panel):
    return all_genes.loc[:, panel if procedure == "thesis_lr" else all_genes.columns]


def fit_safe(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    serious = [str(item.message) for item in caught if "converge" in str(item.message).lower()]
    if serious:
        raise RuntimeError("Model did not converge: " + serious[0])
    return model
