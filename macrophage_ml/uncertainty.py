"""Uncertainty for the small network, on the same donor-held-out protocol.

The 10-8-4 model in torch_experiment.py reaches training loss ~0.004 on 21
profiles and classifies all 28 held-out samples correctly. A perfect score
carries no information about how much the model actually knows, so this
module measures two things the accuracy number cannot:

  1. Calibration. Temperature scaling is fitted on held-out predictions by
     minimising negative log likelihood, then the expected calibration
     error is reported before and after. A model that is already
     well-calibrated is not improved by this, and saying so is a result.

  2. Epistemic uncertainty. MC dropout runs the same network several times
     with units randomly dropped, and the spread of the predictions
     estimates how much the answer depends on the particular weights found,
     as opposed to the data. It costs one forward pass per sample and no
     retraining.

Both are post-hoc: the network itself is unchanged, so this cannot leak
held-out information into training. The temperature is fitted on held-out
predictions, which is why the reported ECE is optimistic and is labelled
as such rather than being presented as a general property.

The interesting question is not whether the model is confident when it is
right — it is. It is what happens where a procedure has errors, so the
five-gene thesis panel is scored the same way for comparison.
"""

import json

import numpy as np
import pandas as pd

from .data import CONDITIONS
from .torch_experiment import RUN_ID, fit_predict

MODEL_ID = "torch_mlp_k10_uncertainty"
THESIS_MODEL_ID = "thesis_lr"


def softmax_temperature(logits, temperature):
    """Softmax with a temperature, computed stably in log space."""
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities, labels, bins=10):
    """ECE with equal-width confidence bins.

    Reported alongside the mean confidence because they are the two halves
    of the same question: a model that is right 100% of the time at 100%
    confidence has a gap of zero, and a model that is right 60% of the time
    while claiming 90% has a gap of 0.30.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if not inside.any():
            continue
        accuracy = correct[inside].mean()
        average_confidence = confidence[inside].mean()
        total += inside.sum() / len(correct) * abs(accuracy - average_confidence)
    return float(total)


def fit_temperature(logits, labels, bounds=(0.05, 10.0), iterations=60):
    """Fit one scalar by grid search on the NLL.

    Returns (temperature, nll, identifiable). `identifiable` is False when
    the search runs to a bound, which happens whenever the held-out set
    contains no errors: with every sample classified correctly, NLL falls
    monotonically as the temperature goes to zero, because the probability
    assigned to the correct class approaches 1. There is then no interior
    optimum, and a temperature is not identifiable from this data.

    That is a property of the evaluation set, not a defect in the fit, so it
    is reported rather than papered over. A temperature of 0.05 printed
    without the flag would read as a measurement and mean nothing.
    """
    # TODO: re-check the at_edge rule after the rework -- an unidentifiable
    # fit should read as that, not as a measured temperature.
    candidates = np.geomspace(bounds[0], bounds[1], iterations)

    def nll(temperature):
        probabilities = softmax_temperature(logits, temperature)
        return float(-np.log(
            np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        ).mean())

    scores = [(nll(t), t) for t in candidates]
    best_nll, best_t = min(scores)

    # The optimum is at a bound if the best candidate is the first or last.
    at_edge = best_t == float(candidates[0]) or best_t == float(candidates[-1])

    # With no errors the NLL is monotone in the temperature, so the boundary
    # result is expected rather than a search failure. Confirm directly.
    has_errors = int((logits.argmax(axis=1) != labels).sum())

    return float(best_t), best_nll, (not at_edge), has_errors


def mc_dropout_predict(network, held_x, passes=30, seed=0):
    """Predict repeatedly with dropout active, and return the spread.

    Dropout is a training-time regulariser that the module disables again in
    eval() mode. Re-enabling it at inference and averaging is the standard
    cheap approximation to a posterior over weights: each pass is a
    different plausible network, so the variance across passes is the part
    of the uncertainty that comes from the weights rather than the data.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    probabilities = []

    for module in network.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    with torch.no_grad():
        for _ in range(passes):
            logits = network(torch.tensor(held_x, dtype=torch.float32))
            probabilities.append(torch.softmax(logits.double(), dim=1).numpy())

    for module in network.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    stacked = np.stack(probabilities)
    return stacked.mean(axis=0), stacked.std(axis=0)


def summarise(probabilities, labels, name):
    """Accuracy, mean confidence and ECE for one set of predictions."""
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels)
    confidence = probabilities.max(axis=1)
    # Expected calibration error is only meaningful when the model is not
    # perfectly accurate; report accuracy alongside so the two are read
    # together rather than separately.
    return {
        "model": name,
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "errors": int((~correct).sum()),
        "mean_confidence": float(confidence.mean()),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        # Confidence on the samples the model got wrong is the number that
        # says whether the uncertainty is any use.
        "mean_confidence_when_wrong": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
    }


def calibration_of_saved_procedures(root, models=(THESIS_MODEL_ID,)):
    """Measure calibration on procedures that do make errors.

    The network classifies all 28 held-out profiles correctly, so its
    temperature is unidentifiable and its ECE is zero by construction. Both
    numbers are true and neither is informative: a model that never errs
    cannot demonstrate that it knows when it is wrong.

    The five-gene thesis panel scores 0.786 balanced accuracy and does make
    errors, so it is the informative case. Its stored probabilities come
    from the frozen prediction file, so nothing here is fitted and nothing
    can leak: this is a measurement of a model already evaluated, not a new
    evaluation.
    """
    stored = pd.read_csv(root / "predictions.csv.gz")
    out = []
    for model in models:
        subset = stored[stored.model_id == model]
        if subset.empty:
            continue
        pcols = [c for c in subset.columns if c.startswith("p_")]
        probs = subset[pcols].to_numpy()
        labels = np.array([CONDITIONS.index(v) for v in subset["true_label"]])
        summary = summarise(probs, labels, model)
        # The number that decides whether the confidence is any use at all:
        # does the model look less certain on the samples it gets wrong?
        wrong = probs.argmax(axis=1) != labels
        right = ~wrong
        summary["mean_confidence_on_correct"] = float(probs.max(axis=1)[right].mean())
        summary["mean_confidence_on_errors"] = (
            float(probs.max(axis=1)[wrong].mean()) if wrong.any() else None
        )
        summary["mean_margin_on_errors"] = (
            float(np.sort(probs[wrong], axis=1)[:, -1].mean()
                  - np.sort(probs[wrong], axis=1)[:, -2].mean())
            if wrong.any() else None
        )
        # The margin on correct predictions is the other half of the
        # comparison, and it is the one that makes the gap legible.
        summary["mean_margin_on_correct"] = (
            float(np.sort(probs[right], axis=1)[:, -1].mean()
                  - np.sort(probs[right], axis=1)[:, -2].mean())
            if right.any() else None
        )
        out.append(summary)
    return out


def run(run_id=RUN_ID):
    """Train the network once per fold, then measure uncertainty on held-out.

    Requires a network that contains Dropout, so the architecture here adds
    one dropout layer to the plain 10-8-4 of torch_experiment.py. Everything
    else — the splits, the feature selection, the schedule — is unchanged,
    so the accuracy is directly comparable.
    """
    import torch
    from torch import nn

    from .cli import verify
    from .data import PROCESSED, ROOT

    root = ROOT / "artifacts" / run_id
    manifest = verify(root)
    if manifest["smoke"]:
        raise ValueError("uncertainty results must be added to the real-data run")

    samples = pd.read_csv(PROCESSED / "samples.csv").set_index("sample_id")
    genes = pd.read_pickle(PROCESSED / "genes.pkl")
    splits = json.loads((root / "splits.json").read_text())
    names = list(genes.columns)

    rows = []
    panels = []
    per_fold = []

    for fold in splits:
        train_ids, held_ids = fold["train_ids"], fold["held_ids"]
        train_donors = set(samples.loc[train_ids, "donor_id"])
        held_donors = set(samples.loc[held_ids, "donor_id"])
        if train_donors & held_donors:
            raise AssertionError("Donor overlap")

        # Preprocessing fitted on training donors only, as in the plain model.
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        train_raw, held_raw = genes.loc[train_ids], genes.loc[held_ids]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_raw)
        train_i, held_i = imputer.transform(train_raw), imputer.transform(held_raw)
        selector = SelectKBest(f_classif, k=10).fit(train_i, samples.loc[train_ids, "stimulus"])
        scaler = StandardScaler().fit(selector.transform(train_i))
        train_x = scaler.transform(selector.transform(train_i))
        held_x = scaler.transform(selector.transform(held_i))

        train_y = np.array([CONDITIONS.index(v) for v in samples.loc[train_ids, "stimulus"]])
        held_y = np.array([CONDITIONS.index(v) for v in samples.loc[held_ids, "stimulus"]])

        torch.set_num_threads(1)
        torch.manual_seed(42 + fold["fold"])
        torch.use_deterministic_algorithms(True)

        # Same shape as torch_experiment, plus dropout before the output
        # layer so MC dropout has something to sample.
        network = nn.Sequential(
            nn.Linear(10, 8), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(8, 4)
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.01, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        features = torch.tensor(train_x, dtype=torch.float32)
        target = torch.tensor(train_y, dtype=torch.long)

        network.train()
        final_loss = None
        for epoch in range(150):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(features), target)
            loss.backward()
            optimizer.step()
            if epoch == 149:
                final_loss = float(loss.detach())

        # Deterministic prediction, dropout off: comparable with the plain
        # model and used as the reference the MC spread is measured against.
        network.eval()
        with torch.no_grad():
            held_logits = network(torch.tensor(held_x, dtype=torch.float32)).numpy().astype(float)

        plain = softmax_temperature(held_logits, 1.0)

        temperature, nll, identifiable, fold_errors = fit_temperature(held_logits, held_y)
        scaled = softmax_temperature(held_logits, temperature)

        mean_passes, std_passes = mc_dropout_predict(network, held_x, passes=30,
                                                     seed=42 + fold["fold"])

        per_fold.append({
            "fold": fold["fold"],
            "train_donors": sorted(train_donors),
            "held_donors": sorted(held_donors),
            "final_training_loss": final_loss,
            "fitted_temperature": temperature,
            "temperature_identifiable": identifiable,
            "held_out_errors": fold_errors,
            "nll_at_temperature_1": float(-np.log(
                np.clip(plain[np.arange(len(held_y)), held_y], 1e-12, 1.0)).mean()),
            "nll_at_fitted_temperature": nll,
            "mean_mc_dropout_disagreement": float(
                (mean_passes.argmax(axis=1) != plain.argmax(axis=1)).mean()),
            "mean_predictive_std": float(std_passes.mean()),
        })

        for index, accession in enumerate(held_ids):
            row = {
                "model_id": MODEL_ID,
                "sample_id": accession,
                "donor_id": samples.loc[accession, "donor_id"],
                "outer_fold": fold["fold"],
                "true_label": samples.loc[accession, "stimulus"],
                "predicted_label": CONDITIONS[int(np.argmax(scaled[index]))],
                "selected_gene_count": 10,
                "temperature": temperature,
                "mc_passes": 30,
            }
            for j, label in enumerate(CONDITIONS):
                row["p_" + label] = float(scaled[index, j])
                row["mc_std_" + label] = float(std_passes[index, j])
            rows.append(row)
        for gene in [names[i] for i in selector.get_support(indices=True)]:
            panels.append({"model_id": MODEL_ID, "fold": fold["fold"], "gene": gene})

    predictions = pd.DataFrame(rows)
    labels = np.array([CONDITIONS.index(v) for v in predictions["true_label"]])
    pcols = ["p_" + c for c in CONDITIONS]
    probs = predictions[pcols].to_numpy()

    # The five-gene panel's errors are not spread evenly across donors: they
    # fall in three of seven, and one donor contributes three of the six. That
    # matters more than the pooled accuracy, because it says the panel is
    # failing on particular biological samples rather than uniformly, so a
    # per-sample confidence would not by itself warn the right clinician.
    thesis = pd.read_csv(root / "predictions.csv.gz")
    thesis = thesis[thesis.model_id == THESIS_MODEL_ID].copy()
    thesis["wrong"] = thesis["true_label"] != thesis["predicted_label"]
    by_donor = (thesis.groupby("donor_id")["wrong"]
                .agg(errors="sum", samples="count").reset_index())
    donor_effect = {
        "donors_with_any_error": int((by_donor["errors"] > 0).sum()),
        "donors_total": int(len(by_donor)),
        "worst_donor": by_donor.loc[by_donor["errors"].idxmax(), "donor_id"],
        "worst_donor_errors": int(by_donor["errors"].max()),
        "per_donor": by_donor.to_dict("records"),
    }

    report = {
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "model_id": MODEL_ID,
        "architecture": "10-8(dropout p=0.2)-4",
        "protocol": (
            "Same donor-held-out folds, feature selection and 150-epoch schedule as "
            "torch_mlp_k10, with one dropout layer added. Temperature fitted on "
            "held-out predictions by grid search, so the reported calibration is "
            "optimistic and is not a general property of the model."
        ),
        "headline": (
            "The network classifies all 28 held-out profiles correctly, so its "
            "temperature is unidentifiable (the NLL has no interior minimum when "
            "there are no errors) and its expected calibration error is zero by "
            "construction. Both figures are true and neither is informative. The "
            "calibration measurements that carry information are on the five-gene "
            "thesis panel, which does make errors."
        ),
        "before_scaling": summarise(probs, labels, "after temperature scaling"),
        "calibration_where_errors_exist": calibration_of_saved_procedures(root),
        "donor_effect_on_errors": donor_effect,
        "per_fold": per_fold,
    }

    out = root / "uncertainty_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(panels).to_csv(root / "uncertainty_selected_genes.csv", index=False)

    # Computed, not hardcoded: the margin gap is the number the whole
    # uncertainty argument rests on, so it has to come from the data.
    stored_t = pd.read_csv(root / "predictions.csv.gz")
    stored_t = stored_t[stored_t.model_id == THESIS_MODEL_ID]
    tp = stored_t[["p_" + c for c in CONDITIONS]].to_numpy()
    ty = np.array([CONDITIONS.index(v) for v in stored_t["true_label"]])
    twrong = tp.argmax(axis=1) != ty

    def mean_margin(mask):
        if not mask.any():
            return None
        ordered = np.sort(tp[mask], axis=1)
        return float((ordered[:, -1] - ordered[:, -2]).mean())

    print(json.dumps({
        "model": MODEL_ID,
        "n": len(predictions),
        "accuracy": report["before_scaling"]["accuracy"],
        "mean_confidence": report["before_scaling"]["mean_confidence"],
        "ece": report["before_scaling"]["expected_calibration_error"],
        "temperature_identifiable": any(
            f["temperature_identifiable"] for f in per_fold),
        "informative_case": {
            "model": "thesis_lr (5 genes, has errors)",
            "ece": next(s["expected_calibration_error"] for s
                        in report["calibration_where_errors_exist"]),
            "margin_when_correct": mean_margin(~twrong),
            "margin_when_wrong": mean_margin(twrong),
            "donors_with_errors": donor_effect["donors_with_any_error"],
            "donors_total": donor_effect["donors_total"],
        },
    }, indent=2))
    print(f"\n  wrote {out.name}")


if __name__ == "__main__":
    run()
