"""A small PyTorch baseline on the same held-out donors as the linear models.

The architecture and training schedule are fixed before evaluation. Feature
selection, imputation and scaling are fitted on outer-training donors only.
"""

from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data import CONDITIONS, PROCESSED, ROOT, digest


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
