# Data audit: GSE46903

The audit and the task definition were recorded on 29 September 2026 before training. GEO's series matrix has 384 arrays, of which 299 are described as macrophages. The selected tissue preparation is GM-CSF differentiated macrophages, collected at 72 hours on platform GPL6947. Four conditions have 34 eligible profiles from 13 exact donor IDs. Requiring one profile of each condition per donor leaves seven donors and 28 profiles, seven per class. The other six eligible donors lack at least one chosen condition or contribute an unmatched sample; they are not included in the primary task.

| Condition as deposited | Selected profiles | Distinct selected donors |
| --- | ---: | ---: |
| `con` | 7 | 7 |
| `IFNg` | 7 | 7 |
| `IL4` | 7 | 7 |
| `TNFa+PGE2+P3C` | 7 | 7 |

Selected donor IDs are `BC11`, `BC13`, `BC52`, `BC58`, `BC76`, `BC78`, and `BC9`. Their sample accessions and exact assignment appear in `artifacts/gse46903_34b8b8ec86_s42/task_definition.json` and the generated `data/processed/samples.csv`. The latter can be recreated with `python -m macrophage_ml.cli prepare`. Exact string matching is used; unverified aliases are never merged.

Source files and SHA-256:

| Official file | Size | SHA-256 |
| --- | ---: | --- |
| [GSE46903 series matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE46nnn/GSE46903/matrix/GSE46903_series_matrix.txt.gz) | 53,717,800 bytes | `2f97da65f77cdf46bd3d73d85a34282acbffc33ee879bcbbb53f2bd740499e0f` |
| [GPL6947 annotation](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/annot/GPL6947.annot.gz) | 6,956,785 bytes | `244c4ef6394abd8c87309318f3f27e9e63c45d2d1c3e89303818f1f63a290824` |

The matrix has 47,463 probes per sample. The GPL6947 table identifies an Illumina HumanHT-12 v3 beadchip. The analysis discards probes with ambiguous or missing gene symbols and takes a within-sample median for probes mapping to one gene. This gives 19,235 mapped genes for the 28 profiles, with no missing cells in the resulting matrix. All five thesis-related panel symbols are present. No outcome-based probe selection precedes the folds.

GEO describes the values as log2 transformed and quantile-normalized in Partek. It does not provide a training-only normalization reference for these chosen folds. The upstream normalization may have used held-out arrays, and the project reports that limitation explicitly. It does not present the result as a fresh-donor prospective test of raw assay processing. The label is the deposited stimulation condition. There is no disease or patient outcome in this task.
