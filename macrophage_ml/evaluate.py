"""Nested donor-grouped evaluation. No operation learns from a held-out donor."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CONDITIONS, PANEL, PROCESSED, ROOT, digest


class TopVariance(BaseEstimator, TransformerMixin):
    def __init__(self, k=1500):
        self.k = k

    def fit(self, X, y=None):
        variances = np.nanvar(X, axis=0)
        self.indices_ = np.sort(np.argsort(-variances, kind="stable")[:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return X[:, self.indices_]


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


def choose(procedure, X, y, donors, seed):
    if procedure == "dummy": return {}, None
    trials = []
    inner = group_folds(donors, 2, seed)
    for params in candidates(procedure):
        predicted = np.empty(len(y), dtype=object)
        valid = True
        for train, held in inner:
            model = build(procedure, params)
            try:
                fit_safe(model, X.iloc[train], y[train])
                predicted[held] = model.predict(X.iloc[held])
            except (ValueError, RuntimeError):
                valid = False
                break
        if valid:
            score = balanced_accuracy_score(y, predicted)
            trials.append((score, params))
    if not trials:
        raise RuntimeError(f"No valid inner fits for {procedure}")
    # A fixed tie rule favours smaller panels and stronger regularization.
    score, params = max(trials, key=lambda pair: (pair[0], -pair[1].get("k", 0),
                             -pair[1].get("C", 0), pair[1].get("l1_ratio", 0)))
    return params, float(score)


def selection(model, procedure, columns):
    if procedure == "dummy": return []
    if "select" in model.named_steps:
        selector = model.named_steps["select"]
        indices = selector.get_support(indices=True) if hasattr(selector, "get_support") else selector.indices_
        return list(np.asarray(columns)[indices])
    return list(columns)


def explanations(model, procedure, X, selected, proba, classes):
    if procedure == "dummy" or procedure == "compact_gb": return [None] * len(X)
    imputed = model.named_steps["impute"].transform(X)
    if "select" in model.named_steps:
        imputed = model.named_steps["select"].transform(imputed)
    scaled = model.named_steps["scale"].transform(imputed)
    linear = model.named_steps["model"]
    out = []
    for row, probabilities in zip(scaled, proba, strict=True):
        top, other = np.argsort(probabilities)[-2:][::-1]
        weights = linear.coef_[top] - linear.coef_[other]
        baseline = float(linear.intercept_[top] - linear.intercept_[other])
        contributions = row * weights
        observed = baseline + float(np.sum(contributions))
        expected = float(np.log(probabilities[top] / probabilities[other]))
        if not np.isclose(observed, expected, atol=1e-5):
            raise AssertionError("Log-odds explanation does not reconcile")
        strongest = np.argsort(-np.abs(contributions))[:5]
        out.append({"relative_to": str(classes[other]), "intercept": baseline,
                    "top_terms": [{"gene": str(selected[i]), "log_odds": float(contributions[i])} for i in strongest],
                    "other_genes": float(np.sum(contributions) - np.sum(contributions[strongest])),
                    "total_log_odds": observed})
    return out


def summaries(predictions, procedures, seed):
    rng = np.random.default_rng(seed + 1000)
    results = {}
    for name in procedures:
        x = predictions[predictions.model_id.eq(name)].copy()
        true = x.true_label.to_numpy()
        pred = x.predicted_label.to_numpy()
        proba = x[["p_" + label for label in CONDITIONS]].to_numpy()
        matrix = confusion_matrix(true, pred, labels=CONDITIONS).tolist()
        donor_balanced = np.mean([
            np.mean([(part.predicted_label == label).mean() for _, part in x[x.true_label.eq(label)].groupby("donor_id")])
            for label in CONDITIONS])
        donor_ids = sorted(x.donor_id.unique())
        donor_indices = {donor: np.flatnonzero(x.donor_id.to_numpy() == donor) for donor in donor_ids}
        boot = []
        for _ in range(2000):
            sampled = rng.choice(donor_ids, len(donor_ids), replace=True)
            picked = np.concatenate([donor_indices[donor] for donor in sampled])
            boot.append(balanced_accuracy_score(true[picked], pred[picked]))
        sorted_labels = sorted(CONDITIONS)
        sorted_proba = x[["p_" + label for label in sorted_labels]].to_numpy()
        results[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=CONDITIONS, average="macro", zero_division=0)),
            "log_loss": float(log_loss(true, sorted_proba, labels=sorted_labels)),
            "donor_balanced_recall": float(donor_balanced),
            "confusion": matrix, "support": [int(sum(true == label)) for label in CONDITIONS],
            "donor_bootstrap_95_interval": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "interval_note": "Approximate donor bootstrap conditional on these held-out predictions; no model refitting",
        }
    return results


def run(seed=42, quick=False):
    task = json.loads((PROCESSED / "task_definition.json").read_text())
    samples = pd.read_csv(PROCESSED / "samples.csv").set_index("sample_id")
    genes = pd.read_pickle(PROCESSED / "genes.pkl")
    if list(genes.index) != task["sample_ids"] or list(samples.index) != task["sample_ids"]:
        raise AssertionError("Matrix and metadata sample order disagree")
    donors = samples.donor_id.to_numpy()
    y = samples.stimulus.to_numpy()
    if set(y) != set(CONDITIONS): raise ValueError("Incorrect task classes")
    outer = group_folds(donors, 3, seed)
    procedures = ["dummy", "thesis_lr", "compact_lr", "compact_lr_k5", "compact_lr_k10",
                  "compact_lr_k20", "compact_lr_k50"]
    if not quick: procedures.extend(["broad_lr", "compact_gb"])
    split_manifest, records, selected_records, explanation_records = [], [], [], []
    for fold, (train, held) in enumerate(outer):
        fold_info = {"fold": fold, "train_donors": sorted(set(donors[train])), "held_donors": sorted(set(donors[held])),
                     "train_ids": list(genes.index[train]), "held_ids": list(genes.index[held])}
        if set(fold_info["train_donors"]) & set(fold_info["held_donors"]): raise AssertionError("Donor overlap")
        split_manifest.append(fold_info)
        for procedure in procedures:
            print(f"Outer fold {fold + 1}/{len(outer)}: {procedure}", flush=True)
            X = matrix_for(procedure, genes, task["reference_panel"])
            params, inner_score = choose(procedure, X.iloc[train], y[train], donors[train], seed + fold)
            model = fit_safe(build(procedure, params), X.iloc[train], y[train])
            labels = model.classes_ if procedure == "dummy" else model.named_steps["model"].classes_
            scores = model.predict_proba(X.iloc[held])
            ordered = np.column_stack([scores[:, list(labels).index(label)] for label in CONDITIONS])
            predicted = np.array(CONDITIONS)[np.argmax(ordered, axis=1)]
            genes_selected = selection(model, procedure, X.columns)
            explanation = explanations(model, procedure, X.iloc[held], genes_selected, scores, labels)
            for gene in genes_selected:
                if procedure not in ("dummy", "broad_lr"):
                    selected_records.append({"model_id": procedure, "fold": fold, "gene": gene})
            for i, row_index in enumerate(held):
                accession = str(genes.index[row_index])
                record = {"model_id": procedure, "sample_id": accession, "donor_id": str(donors[row_index]),
                          "outer_fold": fold, "true_label": str(y[row_index]), "predicted_label": str(predicted[i]),
                          "selected_gene_count": len(genes_selected), "inner_balanced_accuracy": inner_score,
                          "params": json.dumps(params, sort_keys=True)}
                record.update({"p_" + label: float(ordered[i, j]) for j, label in enumerate(CONDITIONS)})
                if not np.isclose(ordered[i].sum(), 1.0): raise AssertionError("Probabilities fail normalization")
                records.append(record)
                if explanation[i] is not None:
                    explanation_records.append({"model_id": procedure, "sample_id": accession, "details": explanation[i]})
    predictions = pd.DataFrame(records)
    for procedure in procedures:
        group = predictions[predictions.model_id.eq(procedure)]
        if len(group) != len(samples) or group.sample_id.nunique() != len(samples):
            raise AssertionError("Incomplete or duplicate out-of-fold predictions")
    metrics = summaries(predictions, procedures, seed)
    run_id = "gse46903_" + task["task_hash"][:10] + f"_s{seed}" + ("_smoke" if quick else "")
    output = ROOT / "artifacts" / run_id
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(selected_records).to_csv(output / "selected_genes.csv", index=False)
    (output / "explanations.json").write_text(json.dumps(explanation_records, indent=2) + "\n")
    (output / "splits.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (output / "task_definition.json").write_text(json.dumps(task, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps({"schema_version": 1, "classes": CONDITIONS,
                           "primary_model": "compact_lr", "results": metrics}, indent=2) + "\n")
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed, "smoke": quick, "task_hash": task["task_hash"], "n_samples": len(samples),
                "n_donors": len(set(donors)), "procedures": procedures, "source_hashes": task["source_hashes"],
                "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    import sklearn
    manifest["versions"]["sklearn"] = sklearn.__version__
    manifest["files"] = {file.name: digest(file) for file in output.iterdir() if file.is_file()}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output, metrics
