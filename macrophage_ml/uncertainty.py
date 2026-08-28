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
