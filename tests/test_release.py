import json
import os
from pathlib import Path
import unittest

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from macrophage_ml.cli import verify
from macrophage_ml.data import CONDITIONS, ROOT


RUN = "gse46903_34b8b8ec86_s42"
PATH = ROOT / "artifacts" / RUN


class ReleaseTests(unittest.TestCase):
    def test_recompute_metrics_and_donor_isolation(self):
        manifest = verify(PATH)
        metrics = json.loads((PATH / "metrics.json").read_text())["results"]
        predictions = pd.read_csv(PATH / "predictions.csv.gz")
        self.assertEqual(set(predictions.model_id), set(manifest["procedures"]))
        for name, group in predictions.groupby("model_id"):
            self.assertEqual(len(group), 28)
            self.assertEqual(group.sample_id.nunique(), 28)
            score = balanced_accuracy_score(group.true_label, group.predicted_label)
            matrix = confusion_matrix(group.true_label, group.predicted_label, labels=CONDITIONS).tolist()
            self.assertAlmostEqual(score, metrics[name]["balanced_accuracy"])
            self.assertEqual(matrix, metrics[name]["confusion"])
        splits = json.loads((PATH / "splits.json").read_text())
        held_ids = []
        for fold in splits:
            self.assertFalse(set(fold["train_donors"]) & set(fold["held_donors"]))
            held_ids.extend(fold["held_ids"])
            group = predictions[predictions.outer_fold.eq(fold["fold"])]
            self.assertEqual(set(group.sample_id), set(fold["held_ids"]))
            self.assertEqual(set(group.donor_id), set(fold["held_donors"]))
        self.assertEqual(len(held_ids), len(set(held_ids)))

if __name__ == "__main__":
    unittest.main()
