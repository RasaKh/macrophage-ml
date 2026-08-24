# Implementation status · 29 September 2026

The first real-data version runs end to end. It downloads the GEO matrix and platform annotation, selects the matched cohort, maps probes to genes, evaluates nine procedures with donor-held-out folds, saves individual predictions and serves a read-only Flask walkthrough. The package includes the tested result bundle, source code, theme files and dependencies. It has not been deployed to a public website.

The included cohort has **28 expression profiles, seven donors and four conditions**. The primary compact logistic procedure scored **1.00 balanced accuracy** on its pooled out-of-fold predictions. The fixed five-gene thesis panel scored **0.786** and the dummy scored **0.250**. Read `ANALYSIS_REPORT.md` for the context and the upstream-normalization limitation. The perfect result should never be quoted without its sample size, donor grouping and data caveat.

Completed checks: official-source SHA-256 recording, exact accession joins, donor matching, zero donor overlap in stored outer and inner fold construction, train-fold-only feature selection and scaling, one held-out prediction per sample and procedure, independent metric recomputation from the prediction file, signed model contributions that reconcile to logistic log odds, artifact hash verification, Flask response/error checks, and JavaScript syntax checking. `python -m unittest discover -s tests -v` passes.

The implementation narrows the original handoff in these ways:

- The first task uses only seven fully matched donors instead of pooling all 34 eligible 72-hour profiles. This avoids condition representation depending on whether an unmatched donor contributed an array.
- The configuration is frozen in code and `task_definition.json`; there is no YAML file or resume mechanism yet.
- The site serves stored out-of-fold predictions. It does not offer a live inference upload, and no fitted model is serialized for user input.
- The uncertainty interval resamples stored donor predictions. With zero observed errors, it is 1.00–1.00 and is not a useful interval for performance on new donors; that limit is stated on the page.
- The full nested comparison is present, but figure generation, systematic sample-level biological interpretation and a second independent dataset remain further work.
- Browser screenshot comparison was unavailable in this workspace. Flask HTML/API responses and the asset files were checked; visual inspection on desktop and mobile remains necessary before publishing.

The next useful study is a transfer test on a separate macrophage expression dataset with comparable stimulation labels and preprocessing. A panel chosen from this run would need to be frozen before that test. Adding a neural network to these 28 profiles would not be a meaningful way to address the PROMETHEUS vacancy's PyTorch requirement.
