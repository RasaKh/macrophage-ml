# Implementation status · 29 September 2026

The pipeline runs end to end on real data. It downloads the GEO matrix and platform annotation, selects the matched cohort, maps probes to genes, evaluates nine procedures with donor-held-out folds and saves every individual prediction. A read-only walkthrough of the results is at https://macrophage-ml.khosrofar.me.

The included cohort has 28 expression profiles, seven donors and four conditions. The primary compact logistic procedure scored 1.00 balanced accuracy on its pooled out-of-fold predictions, the fixed five-gene thesis panel 0.786 and the dummy 0.250. `ANALYSIS_REPORT.md` gives the context and the upstream-normalization limitation. The perfect result should never be quoted without its sample size, donor grouping and that caveat.

Checks completed: official-source SHA-256 recording, exact accession joins, donor matching, zero donor overlap in stored outer and inner fold construction, train-fold-only feature selection and scaling, one held-out prediction per sample and procedure, independent metric recomputation from the prediction file, signed model contributions that reconcile to logistic log odds, artifact hash verification, Flask response/error checks, and JavaScript syntax checking. `python -m unittest discover -s tests -v` passes.

Scope decisions:

- The first task uses only seven fully matched donors instead of pooling all 34 eligible 72-hour profiles, so that condition representation does not depend on whether an unmatched donor contributed an array.
- The configuration is frozen in code and `task_definition.json`; there is no YAML file or resume mechanism yet.
- The walkthrough serves stored out-of-fold predictions. It does not offer live inference on uploaded data, and no fitted model is serialized for user input.
- The uncertainty interval resamples stored donor predictions. With zero observed errors it is 1.00–1.00, which is not a useful interval for performance on new donors; the walkthrough states that limit.
- Systematic sample-level biological interpretation and a second independent dataset remain further work.

The next useful study is a transfer test on a separate macrophage expression dataset with comparable stimulation labels and preprocessing. A panel chosen from this run would need to be frozen before that test.
