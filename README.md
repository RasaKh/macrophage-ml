# Macrophage stimulation from gene expression

Donor-held-out evaluation of regularized logistic regression, shallow boosted trees and a small PyTorch network on human macrophage gene expression profiles from GEO `GSE46903` (platform `GPL6947`).

A read-only walkthrough of the predictions, gene panels, training curves and caveats is hosted at [macrophage-ml.khosrofar.me](https://macrophage-ml.khosrofar.me).

## Task and data

GEO `GSE46903` contains 384 arrays on an Illumina HumanHT-12 v3 beadchip (`GPL6947`), 299 of them labelled as macrophages. Filtering to GM-CSF differentiated macrophages collected at 72 hours across four stimulation conditions (`con`, `IFNg`, `IL4`, `TNFa+PGE2+P3C`) leaves 34 eligible profiles from 13 donor IDs. Requiring all four conditions per donor yields a matched cohort of 7 donors and 28 profiles (7 per class).

Probes with missing or ambiguous gene symbols are dropped. Multiple probes mapping to one symbol are combined by within-sample median, which gives 19,235 mapped genes with no missing cells. Exact accession lists and file SHA-256 digests are in [DATA_AUDIT.md](DATA_AUDIT.md).

## Evaluation design

Models are evaluated with three outer donor-grouped folds (holding out two or three donors per fold) and two inner donor-grouped folds for hyperparameter selection. No donor appears in both training and held-out sets in any outer or inner split. Median imputation (as a pipeline safeguard), ANOVA F-score feature ranking and scaling are fitted inside training folds only.

A five-gene panel from the author's M.D. thesis (`ABCA1`, `ABCG1`, `NR1H3`, `TNF`, `IL1B`) is a fixed benchmark alongside data-driven ranked panels using regularized logistic regression and shallow gradient-boosted trees.

The repository includes stored artifacts from run `gse46903_34b8b8ec86_s42` (29 September 2026) under `artifacts/gse46903_34b8b8ec86_s42/`:

| Procedure | Balanced accuracy | Errors / 28 |
| --- | ---: | ---: |
| Training-prior dummy | 0.250 | 21 |
| Fixed five-gene thesis panel, logistic regression | 0.786 | 6 |
| Compact ANOVA-ranked panel (size chosen in inner folds), logistic regression | 1.000 | 0 |
| Fixed 5-gene ANOVA-ranked panel, logistic regression | 0.929 | 2 |
| Fixed 10-, 20- or 50-gene ANOVA-ranked panel, logistic regression | 1.000 | 0 |
| Variance-filtered elastic-net logistic regression | 1.000 | 0 |
| Compact ANOVA-ranked panel, shallow boosted trees | 0.929 | 2 |
| Fixed 10-gene PyTorch MLP (`torch_mlp_k10`) | 1.000 | 0 |

The CPU PyTorch baseline (`torch_mlp_k10`) lives in `macrophage_ml/torch_experiment.py`: a fixed 10-8-4 ReLU MLP trained for 150 epochs with full-batch AdamW on ten ANOVA-ranked genes selected inside each outer training fold, with fixed hyperparameters and no inner tuning or early stopping. Its per-fold training curves (`torch_training.json`) and the donor-level bootstrap intervals (`uncertainty_report.json`) sit in `artifacts/gse46903_34b8b8ec86_s42/`.

## Commands

Requires Python 3.10+. Install the analysis dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn` in `requirements-analysis.txt`; add CPU `torch` if re-running `torch_experiment.py`):

```bash
pip install -r requirements-analysis.txt
```

CLI commands (raw and prepared data are written under `data/`; runs write to `artifacts/`):

```bash
python -m macrophage_ml.cli download                                          # fetch GSE46903 matrix and GPL6947 annotation
python -m macrophage_ml.cli audit                                             # print cohort filtering counts
python -m macrophage_ml.cli prepare                                           # map probes to genes and write data/processed/
python -m macrophage_ml.cli evaluate --seed 42                                # run nested CV and write a new artifact folder
python -m macrophage_ml.torch_experiment                                      # run CPU PyTorch baseline and update run artifacts
python -m macrophage_ml.cli verify-release --run-id gse46903_34b8b8ec86_s42   # verify committed artifact hashes and splits offline
```

Run the test suite (runs offline against the committed artifacts):

```bash
python -m unittest discover -s tests -v
```

## Limitations

- Seven matched donors (28 profiles) from a single platform and experimental source are too few to estimate how these panels generalize to a new laboratory or cohort.
- GEO deposited a log2 quantile-normalized matrix produced in Partek. Because held-out arrays may have participated in that upstream normalization, fold-local preprocessing inside this repository cannot undo any leakage that occurred before download.
- Resampling seven donors in the donor bootstrap when the observed error count is zero gives a `1.00-1.00` interval, which describes the observed prediction table rather than uncertainty on unseen donors.
- The labels are in vitro stimulation conditions; there is no patient or disease outcome in this task.

Further documentation in this repository:

- [DATA_AUDIT.md](DATA_AUDIT.md): source URLs, SHA-256 checksums, cohort selection and probe mapping.
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md): per-procedure metrics, error cases and upstream-normalization caveats.
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): scope decisions, completed verification checks and planned follow-up work.
