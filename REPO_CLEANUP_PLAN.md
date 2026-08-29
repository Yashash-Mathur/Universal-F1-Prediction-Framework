# Repository Cleanup & Architecture Plan

## Safety rule

This work is being performed on `repo-cleanup`, branched from `main`.

`main` remains the known-good reference. No production code is deleted or replaced until the replacement has been validated against the current model and the anti-data-leakage rules.

## Current repository map

### Final-model / current pipeline

- `60th_real_pipeline.py` — current historical cleaning + feature-engineering pipeline producing `f1_2020_2026_features_v6.csv`.
- `61_prepare_dutch_gp.py` — race-specific pre-qualifying Dutch GP dataset preparation.
- `62_catboost_backtest.py` — CatBoost historical backtesting.
- `63_catboost_2026_walkforward.py` — 2026 walk-forward validation.
- `64_predict_dutch_gp.py` — final Dutch GP CatBoost prediction flow.
- `65_dutchgp_datafix.py` — Dutch GP qualifying-data correction/injection for the 2026 roster and qualifying session.

These files form the most recent working lineage and must be preserved until the modular replacement is proven equivalent or better.

### Core historical data / feature development

- `mle34_Updated_Collector.py` — important historical FastF1 data collector; candidate for consolidation into the future data layer.
- `mle35_data-checker.py` — data validation/checking.
- `mle36_clean_v2.py` — historical cleaning stage.
- `mle37_feature_engineering_v2.py` — earlier feature-engineering stage.
- `mle38_average_finish.py` — earlier recent-finish feature addition.
- `mle39_teammate_gap.py` — earlier teammate qualifying-gap feature addition.
- `mle44_circuit_type_features.py` — circuit classification feature development.
- `mle45_circuit_features_v2.py` — later circuit-performance feature implementation.
- `mle51_add_recent_form_features.py` — recent-form feature development.
- `mle59_update_raw_dataset.py` — recent raw-dataset update workflow.

These are historical development artifacts and should not be deleted until the final feature pipeline has been audited against them.

### Baseline / model experiments to retain conceptually

- `logistic_regression_mle28.py`
- `mle31_decision_tree.py`
- `mle32_random_forests.py`
- `mle40_baseline_v2.py`
- `mle42_decision_tree_baseline.py`
- `mle43_random_forest.py`
- `mle45_random_forest_v2.py`
- `mle46_catboost_baseline.py`
- `mle47_catboost_time_split.py`
- `mle48_catboost_position_predictor.py`
- `mle49_catboost_stability_test.py`
- `mle50_catboost_stability_test.py`
- `mle51_catboost_v2.py`
- `mle52_catboost_tuning.py`
- `mle52_prequali_catboost.py`
- `mle41_ablation.py`

The final repository should retain a clean baseline/experiments section rather than dozens of root-level scripts. Exact consolidation/deletion decisions require code-level review first.

### EDA / validation

- `mle26(EDA).py` — EDA artifact and should be retained conceptually, preferably as a clean EDA module/notebook.
- `mle33_check.py` — validation/checking.
- `mle35_data-checker.py` — validation/checking.
- `checking_code.py` — later checking/diagnostic script.
- `mle58_prediction_audit.py` — prediction audit.
- `mle7_2022check.py` — historical validation artifact.
- `mle13race_check.py` — historical race-data validation artifact.

### Barcelona / race-specific historical experiments

- `mle53_create_barcelona_prediction_dataset.py`
- `mle55_barcelona_quali_injector.py`
- `mle56_barcelona_final_prediction.py`

These document an earlier race-specific prediction workflow. They are candidates for archival/removal after their useful logic has been absorbed into the generalized race-prediction flow.

### Early development lineage

- `mle1.py` through `mle25.py`
- `mle28.py`
- `mle29.py`
- `mle3.py`, `mle4.py`, `mle5.py`, `mle6.py`, `mle9.py`
- `mle8(main collector).py`
- `train_testdata_split_mle27.py`

These appear to be early iterative development artifacts based on their naming and chronology. They are **not approved for deletion yet**. They must be treated as historical until their code/dependencies have been checked.

### Miscellaneous development files

- `dg1.py`
- `dg2.py`
- `did-data-add.py`

These require code-level inspection before classification.

### Repository support

- `.gitignore` — retain and review.
- `requirements.txt` — retain and later clean/pin only after dependency audit.
- `README.md` — retain and rewrite after architecture is stable.
- `tempCodeRunnerFile.py` — strong deletion candidate; verify it has no project purpose before removal.

## Target architecture

The repository should eventually separate responsibilities without creating one fragile `main.py`:

```text
Universal-F1-Prediction-Framework/
├── data/
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── validation/
│   └── prediction/
├── experiments/
│   ├── eda/
│   ├── baselines/
│   └── model_comparison/
├── outputs/
├── README.md
└── requirements.txt
```

This is a target, not an instruction to move files immediately.

## Data strategy

Do not create a permanent CSV for every transformation.

Preferred future flow:

```text
FastF1/raw data
    -> canonical raw dataset
    -> deterministic feature pipeline
    -> model-ready dataframe
    -> training/backtest/prediction
```

A small number of deliberate persisted artifacts may remain useful, especially raw historical data and reproducible model outputs. Intermediate CSV proliferation should be removed only after reproducibility has been demonstrated.

## Anti-leakage requirements

These are non-negotiable during cleanup:

1. Every race prediction feature must use only information available before that race.
2. Championship points/positions must be pre-race.
3. Recent-form features must exclude the target race.
4. Qualifying features may be used for a post-qualifying podium prediction, but never race-result information.
5. Walk-forward validation must remain chronological.
6. Circuit-history features must remain strictly historical relative to the target race.
7. No cleanup refactor may silently change the feature definitions used by the validated CatBoost model.
8. Any feature removal must be supported by an ablation/backtest comparison before being merged.

## Circuit features

The circuit-performance features are **not being removed at this stage**. They are part of the current v6 feature set and could affect prediction quality. Their future status will be decided only after an ablation comparison using the same leakage-safe chronological evaluation.

## Next safe steps

1. Finish code-level dependency review of the current and historical scripts.
2. Identify the exact lineage of the current v6 feature dataset.
3. Reproduce the current CatBoost backtest/prediction from the untouched baseline branch.
4. Build modular replacements one component at a time.
5. Compare old vs new outputs/features before deleting anything.
6. Move baselines and EDA into clearly named experiment areas.
7. Remove obsolete scripts only after their logic is either preserved or formally archived.
8. Run a final anti-leakage audit and regression test.
9. Open a pull request from `repo-cleanup` to `main` only after validation passes.
