"""Calibration and uncertainty for the small network.

Two post-hoc measurements on the frozen held-out predictions and the trained
network: temperature-scaled calibration and the MC-dropout spread of answers.
"""

import numpy as np

from .data import CONDITIONS


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
    contains no errors.
    """
    candidates = np.geomspace(bounds[0], bounds[1], iterations)

    def nll(temperature):
        probabilities = softmax_temperature(logits, temperature)
        return float(-np.log(
            np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        ).mean())

    scores = [(nll(t), t) for t in candidates]
    best_nll, best_t = min(scores)
    at_edge = best_t == float(candidates[0]) or best_t == float(candidates[-1])
    has_errors = int((logits.argmax(axis=1) != labels).sum())
    return float(best_t), best_nll, (not at_edge), has_errors


def mc_dropout_predict(network, held_x, passes=30, seed=0):
    """Predict repeatedly with dropout active, and return the spread."""
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
    return {
        "model": name,
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "errors": int((~correct).sum()),
        "mean_confidence": float(confidence.mean()),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        "mean_confidence_when_wrong": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
    }
