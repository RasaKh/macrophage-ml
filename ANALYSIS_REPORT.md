# Analysis report

Run `gse46903_34b8b8ec86_s42`, completed 29 September 2026. The result comes from 28 matched profiles across seven donors and four laboratory conditions, using the deposited GSE46903 matrix. Three outer donor-grouped folds each held out two or three donors. Two inner donor-grouped folds selected each procedure's settings. Every profile received one out-of-fold prediction per procedure. No donor appears on both sides of an outer or inner split.

| Procedure | Balanced accuracy | Errors / 28 |
| --- | ---: | ---: |
| Training-prior dummy | 0.250 | 21 |
| Fixed thesis panel, logistic regression | 0.786 | 6 |
| Compact panel, size chosen in inner folds | 1.000 | 0 |
| Fixed 5-gene ranked panel | 0.929 | 2 |
| Fixed 10-gene ranked panel | 1.000 | 0 |
| Fixed 20-gene ranked panel | 1.000 | 0 |
| Fixed 50-gene ranked panel | 1.000 | 0 |
| Variance-filtered elastic-net logistic regression | 1.000 | 0 |
| Compact ranked panel, shallow boosted trees | 0.929 | 2 |

The primary procedure selected 50 genes in two outer folds and 20 in the third. The panels differ by fold. It would be misleading to publish one list as if it were an experimentally established signature. Some repeatedly selected genes, such as CXCL9, CXCL10 and several GBP family members, are plausible in stimulation responses, but this analysis does not show that they cause those responses.

The five-gene thesis-related panel made six errors, including three among donor BC58's four profiles. The five-gene ranked procedure classified two control profiles as IL4. These examples are exposed in the sample explorer; unsuccessful predictions remain available. The compact procedure made none in this cohort. This is an interesting difference between panels, but seven donors are too few to establish how either will perform in a new laboratory or population.

The stored donor bootstrap gives 1.00–1.00 for procedures with no errors because every resample of these seven donors still has no errors. That is a property of this observed prediction table, **not** a tight interval for future performance. The code records this interval only as a conditional descriptive check. We do not infer a clinical-use threshold or use the model's class probabilities as calibrated confidence.

The strongest data limitation precedes our own cross-validation: GEO deposited a log2 quantile-normalized matrix. If quantile normalization included all samples, held-out arrays contributed to the shared preprocessing. Our fold-specific imputation, ranking, scaling and tuning do not undo that. The study is also small, single-platform, from one experimental source and limited to four exposure labels. The model is not trained to classify macrophage biology in patient tissue, measure secreted TNF-alpha/IL-1beta, estimate a treatment effect, or make a clinical decision.

The experiment is reproducible through the source links and checksums in [DATA_AUDIT.md](DATA_AUDIT.md). `splits.json`, `predictions.csv.gz`, `metrics.json`, `selected_genes.csv` and `run_manifest.json` let a reviewer check the outcome rather than rely on a screenshot.
