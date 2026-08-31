"""A small PyTorch baseline on the same held-out donors as the linear models.

The architecture and training schedule are fixed before evaluation. Feature
selection, imputation and scaling are fitted on outer-training donors only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data import CONDITIONS, PROCESSED, ROOT, digest
from .evaluate import summaries


MODEL_ID = "torch_mlp_k10"
RUN_ID = "gse46903_34b8b8ec86_s42"


def fit_predict(train_x, train_y, held_x, seed):
    """Train one fixed small network on CPU and return held-out probabilities."""
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = nn.Sequential(nn.Linear(train_x.shape[1], 8), nn.ReLU(), nn.Linear(8, len(CONDITIONS)))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    features = torch.tensor(train_x, dtype=torch.float32)
    target = torch.tensor(train_y, dtype=torch.long)
    network.train()
    losses = []
    for epoch in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(network(features), target)
        loss.backward()
        optimizer.step()
        if epoch in {0, 9, 24, 49, 99, 149}:
            losses.append({"epoch": epoch + 1, "training_loss": float(loss.detach())})
    network.eval()
    with torch.no_grad():
        raw = network(torch.tensor(held_x, dtype=torch.float32))
        probabilities = torch.softmax(raw.double(), dim=1).numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Invalid PyTorch probability output")
    return probabilities, losses


def run(run_id=RUN_ID):
    import torch
    from .cli import verify

    path = ROOT / "artifacts" / run_id
    manifest = verify(path)
    if manifest["smoke"]:
        raise ValueError("PyTorch results must be added to the real-data run")
    samples = pd.read_csv(PROCESSED / "samples.csv").set_index("sample_id")
    genes = pd.read_pickle(PROCESSED / "genes.pkl")
    task = json.loads((path / "task_definition.json").read_text())
    splits = json.loads((path / "splits.json").read_text())
    if list(genes.index) != task["sample_ids"] or list(samples.index) != task["sample_ids"]:
        raise AssertionError("Processed matrix and frozen task have different sample order")
    names = list(genes.columns)
    y = samples.stimulus.to_numpy()
    new_rows, panels, training = [], [], []
    for fold in splits:
        train_ids, held_ids = fold["train_ids"], fold["held_ids"]
        if set(samples.loc[train_ids, "donor_id"]) & set(samples.loc[held_ids, "donor_id"]):
            raise AssertionError("Donor overlap")
        train_raw, held_raw = genes.loc[train_ids], genes.loc[held_ids]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_raw)
        train_i, held_i = imputer.transform(train_raw), imputer.transform(held_raw)
        selector = SelectKBest(f_classif, k=10).fit(train_i, samples.loc[train_ids, "stimulus"])
        chosen = [names[index] for index in selector.get_support(indices=True)]
        scaler = StandardScaler().fit(selector.transform(train_i))
        train_x = scaler.transform(selector.transform(train_i))
        held_x = scaler.transform(selector.transform(held_i))
        labels = np.array([CONDITIONS.index(value) for value in samples.loc[train_ids, "stimulus"]])
        probabilities, losses = fit_predict(train_x, labels, held_x, 42 + fold["fold"])
        training.append({"fold": fold["fold"], "train_donors": fold["train_donors"],
                         "held_donors": fold["held_donors"], "losses": losses,
                         "selected_genes": chosen})
        for gene in chosen:
            panels.append({"model_id": MODEL_ID, "fold": fold["fold"], "gene": gene})
        for index, accession in enumerate(held_ids):
            prediction = CONDITIONS[int(np.argmax(probabilities[index]))]
            row = {"model_id": MODEL_ID, "sample_id": accession,
                   "donor_id": samples.loc[accession, "donor_id"], "outer_fold": fold["fold"],
                   "true_label": samples.loc[accession, "stimulus"], "predicted_label": prediction,
                   "selected_gene_count": 10, "inner_balanced_accuracy": np.nan,
                   "params": json.dumps({"architecture": "10-8-4", "epochs": 150, "lr": .01,
                                         "weight_decay": .05, "fixed_before_evaluation": True}, sort_keys=True)}
            row.update({"p_" + label: float(probabilities[index, j]) for j, label in enumerate(CONDITIONS)})
            new_rows.append(row)
    original = pd.read_csv(path / "predictions.csv.gz")
    original = original[original.model_id.ne(MODEL_ID)]
    combined = pd.concat([original, pd.DataFrame(new_rows)], ignore_index=True)
    if len(new_rows) != len(samples) or len({r["sample_id"] for r in new_rows}) != len(samples):
        raise AssertionError("PyTorch run lacks one held-out prediction per sample")
    selection = pd.read_csv(path / "selected_genes.csv")
    selection = pd.concat([selection[selection.model_id.ne(MODEL_ID)], pd.DataFrame(panels)], ignore_index=True)
    metric_file = path / "metrics.json"
    metrics = json.loads(metric_file.read_text())
    metrics["results"][MODEL_ID] = summaries(combined, [MODEL_ID], 42)[MODEL_ID]
    (path / "torch_training.json").write_text(json.dumps({"model_id": MODEL_ID, "framework": "PyTorch",
         "torch_version": torch.__version__, "task": "Four-class condition classification from ten selected transcripts",
         "protocol": "Fixed 10-8-4 ReLU MLP, full-batch AdamW, 150 epochs, cross entropy, CPU. No outer-fold model selection.",
         "folds": training}, indent=2) + "\n")
    combined.to_csv(path / "predictions.csv.gz", compression="gzip", index=False)
    selection.to_csv(path / "selected_genes.csv", index=False)
    metric_file.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest["procedures"] = [*dict.fromkeys([*manifest["procedures"], MODEL_ID])]
    manifest["versions"]["torch"] = torch.__version__
    manifest["files"] = {file.name: digest(file) for file in path.iterdir()
                         if file.is_file() and file.name != "run_manifest.json"}
    (path / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": MODEL_ID, "n": len(new_rows),
                      "balanced_accuracy": metrics["results"][MODEL_ID]["balanced_accuracy"],
                      "errors": sum(row["true_label"] != row["predicted_label"] for row in new_rows)}, indent=2))


if __name__ == "__main__":
    run()
