# Code Audit — Repository Cleanup

This audit is being performed on `repo-cleanup`, which is branched from `main`.

## Safety status

- `main` is untouched by the cleanup work.
- No production Python files are deleted or replaced by this audit.
- The current Dutch GP prediction lineage is preserved until modular replacements are validated.
- The anti-data-leakage design is treated as a hard constraint.

## Current working lineage

| File | Current role | Audit decision |
|---|---|---|
| `60th_real_pipeline.py` | Historical cleaning + feature engineering through `features_v6` | KEEP as reference; later refactor into modular preprocessing/features modules |
| `61_prepare_dutch_gp.py` | Pre-qualifying Dutch GP state preparation | KEEP as historical race-flow reference; later generalize |
| `62_catboost_backtest.py` | Multi-race chronological CatBoost backtest | KEEP conceptually; later move to validation/experiments |
| `63_catboost_2026_walkforward.py` | 2026 chronological walk-forward validation | KEEP conceptually; later generalize |
| `64_predict_dutch_gp.py` | Final post-qualifying Dutch GP CatBoost inference | KEEP as current known-good reference until generalized predictor is validated |
| `65_dutchgp_datafix.py` | Dutch GP qualifying roster/data correction | KEEP as race-specific historical artifact for now; qualifying injection should later become a reusable prediction input layer |

## Core data / feature lineage

| File | Finding | Decision |
|---|---|---|
| `mle34_Updated_Collector.py` | Historical FastF1 collector. Loads full qualifying and race sessions and writes checkpoints. Useful lineage, but heavier than necessary for the current feature set. | ARCHIVE after replacement collector is validated |
| `mle35_data-checker.py` | Basic raw-data missing-value checks. | ARCHIVE after reusable validator exists |
| `mle36_clean_v2.py` | Early cleaning stage: target creation, pit-lane handling, qualifying flag. | ARCHIVE; logic belongs in preprocessing module |
| `mle37_feature_engineering_v2.py` | Early championship feature implementation. Chronological per-round calculation is leakage-aware but inefficient. | ARCHIVE after feature module is validated |
| `mle38_average_finish.py` | Earlier `AverageFinishLast5` implementation. Correctly uses prior finishes only. | ARCHIVE; preserve logic in feature module |
| `mle39_teammate_gap.py` | Historical teammate qualifying-gap feature. Uses same-race qualifying data and is suitable for post-qualifying prediction. | ARCHIVE; preserve logic in feature module |
| `mle44_circuit_type_features.py` | First circuit-feature implementation using a rolling historical window. | ARCHIVE; preserve circuit definitions and leakage logic |
| `mle45_circuit_features_v2.py` | Later circuit-performance implementation restricted to 2023+ history. | ARCHIVE after current v6 circuit features are validated |
| `mle51_add_recent_form_features.py` | Adds recent-form features with prior-race filtering. Correct concept, but repeated row-wise filtering is inefficient. | ARCHIVE after vectorized feature module is validated |
| `mle59_update_raw_dataset.py` | 2026 raw-data updater. Reusable concept, but performs full FastF1 session loads. | ARCHIVE after reusable data collector is validated |

## Baseline / model experiments

### Keep conceptually

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

These should eventually live under `experiments/baselines/` or `experiments/model_comparison/`, with clean names and a shared leakage-safe evaluation utility.

### Important validation findings

Several older experiments use random train/test splits. Those are acceptable as historical learning artifacts but must not be presented as the project's primary validation methodology.

`mle47_catboost_time_split.py`, `mle51_catboost_v2.py`, `mle52_catboost_tuning.py`, and the earlier stability scripts use the future test period during model selection or preprocessing in ways that are not suitable for the final anti-leakage evaluation. In particular, whole-dataset median imputation in `mle49_catboost_stability_test.py`, `mle50_catboost_stability_test.py`, and `mle45_random_forest_v2.py` can incorporate future information.

The replacement evaluation layer must fit preprocessing/imputation using training data only and use chronological walk-forward evaluation for model selection.

## EDA / validation

- `mle26(EDA).py` — retain conceptually, but replace with a proper clean EDA notebook/script using the canonical dataset.
- `mle33_check.py` — historical diagnostic scratchpad; archive after reusable validation checks exist.
- `mle35_data-checker.py` — archive after reusable validator exists.
- `checking_code.py` — archive after reusable validator exists.
- `mle58_prediction_audit.py` — retain conceptually and generalize into prediction audit tooling.
- `mle7_2022check.py` — historical debugging artifact; archive.
- `mle13race_check.py` — historical debugging artifact; archive.

## Race-specific historical experiments

- `mle53_create_barcelona_prediction_dataset.py`
- `mle55_barcelona_quali_injector.py`
- `mle56_barcelona_final_prediction.py`

These successfully document the Barcelona prediction workflow but are hard-coded to one event. Their useful logic should be generalized into the prediction input and inference layers, then the scripts can be archived.

## Early development lineage

The following files are early learning/debugging artifacts and should not remain as root-level production scripts:

- `mle1.py` through `mle25.py`
- `mle28.py`
- `mle29.py`
- `mle30clean_dataset.py`
- `mle31_decision_tree.py`
- `mle32_random_forests.py`
- `mle33_check.py`
- `mle4.py`, `mle5.py`, `mle6.py`, `mle9.py`
- `mle8(main collector).py`
- `train_testdata_split_mle27.py`

The code audit confirms these are development/debugging stages rather than required runtime components. They should be archived or removed only after the useful logic is represented in the new structure.

## Miscellaneous

- `dg1.py` — simple 2026 dataset count diagnostic. ARCHIVE.
- `dg2.py` — 2026 schedule-vs-dataset diagnostic. ARCHIVE after reusable validation exists.
- `did-data-add.py` — one-off race-count diagnostic. ARCHIVE.
- `tempCodeRunnerFile.py` — editor-generated scratch file. DELETE on cleanup branch; it has no project role.
- `.gitignore` — KEEP and review later. It currently ignores all CSVs, which keeps generated datasets out of the public repository.
- `requirements.txt` — KEEP; later reduce/review direct dependencies and document the supported Python environment.
- `README.md` — KEEP; rewrite after architecture and evaluation rules are finalized. Current wording describes an older pipeline and needs alignment with the current post-qualifying prediction workflow.

## Critical technical findings

### 1. Circuit features are not to be removed yet

The current v6 model uses four circuit-performance features plus four history flags. Removing them without an ablation comparison could change predictions. The correct optimization is to preserve their definitions first, then benchmark a circuit-free variant using the same chronological walk-forward evaluation.

### 2. The current feature pipeline is computationally heavier than necessary

`60th_real_pipeline.py` calculates circuit history using a row-by-row loop that repeatedly filters the full dataframe. This is an optimization target, not a reason to remove the features. A vectorized/grouped implementation should produce the same feature values and then be regression-tested against v6.

### 3. FastF1 usage should be narrowed

Historical collectors call broad `session.load()` operations. The current model does not require full telemetry for the core dataset. A future collector should request only the session data required to construct the canonical raw dataset and rely on FastF1's cache. This reduces runtime and unnecessary downloads without changing the model definition.

### 4. Intermediate CSV proliferation should stop

The future pipeline should use a small number of deliberate persisted artifacts:

`FastF1 -> canonical raw data -> deterministic feature dataframe -> training/backtest/prediction`

Intermediate feature-version CSVs such as `v2`, `v3`, `v4`, `v5`, and `v6` should no longer be the architecture. The exact persistent artifacts will be decided after the current v6 output is reproduced exactly.

### 5. No single `main.py`

The target architecture remains modular:

```text
data/
src/
  data_collection/
  preprocessing/
  features/
  models/
  validation/
  prediction/
experiments/
  eda/
  baselines/
  model_comparison/
outputs/
```

A thin orchestration layer may call these modules, but the system must not depend on one large fragile script.

## Anti-leakage gate for every refactor

A refactor cannot replace the current implementation until all of the following are demonstrated:

1. Training data for a target race contains only earlier races.
2. Championship points/positions are pre-race.
3. Recent-form features exclude the target race.
4. Circuit-history features exclude the target race and use only permitted historical seasons.
5. Qualifying features are used only for post-qualifying predictions.
6. Any imputation or preprocessing is fitted only on the training portion during evaluation.
7. Walk-forward validation remains chronological.
8. Current v6 feature values can be reproduced within an explicitly defined numerical tolerance.
9. Model ranking/prediction behavior is regression-tested before old code is archived.

## Next implementation order

1. Create the canonical feature/schema contract.
2. Extract the current v6 feature definitions into modular pure-Python functions without changing them.
3. Build a vectorized circuit-history implementation and compare it row-for-row against v6.
4. Build the canonical raw-data collector with minimal FastF1 loading.
5. Build a reusable leakage-safe walk-forward evaluator.
6. Rebuild Logistic Regression, Decision Tree, Random Forest, and CatBoost experiments on the shared evaluator.
7. Generalize the race prediction flow so any target GP can be supplied rather than hard-coding Dutch GP.
8. Add reusable prediction auditing and optional Matplotlib terminal/file visuals.
9. Archive/delete obsolete scripts only after the above regression tests pass.
10. Rewrite the README and open the pull request from `repo-cleanup` to `main`.
