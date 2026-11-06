# Universal F1 Prediction Framework

A modular, leakage-safe machine learning platform for predicting Formula 1 podium finishes (Top 3) using strictly pre-race and post-qualifying data.

---

## 🏎️ Overview

The **Universal F1 Prediction Framework** is an end-to-end ML system that collects race and qualifying timing data, cleans and validates session records, engineers strictly pre-race features, and trains gradient-boosted trees (CatBoost) to rank drivers by podium probability.

### Core Capabilities
- **Modular Pipeline**: Clean separation between data collection, preprocessing, feature engineering, modeling, validation, and inference under `src/`.
- **Strict Anti-Data-Leakage Design**: All championship standings, recent form, and circuit history metrics strictly exclude the target event.
- **Fast Execution**: Vectorized circuit and form computations achieving >370× speedups over naive iteration.
- **Chronological Walk-Forward Validation**: Evaluated race-by-race across the 2026 season without future-data contamination.

---

## 📊 Recorded Walk-Forward Performance (2026 Season, 11 Races)

| Metric | Result | Note |
|---|---|---|
| **Binary Classification Accuracy** | **89.26%** | Driver-level binary classification (Podium vs. Non-Podium) |
| **Precision** | **60.61%** | Positive predictive value |
| **Recall** | **54.55%** | Sensitivity |
| **F1 Score** | **0.5619** | Harmonic mean of precision and recall |
| **Average Correct Podium Picks** | **1.82 / 3** | Mean drivers correctly identified in the Top 3 |
| **Perfect Top-3 Predictions** | **2 / 11** | Races with 3/3 correct podium drivers |
| **Races with ≥ 2 Correct Podium Picks** | **7 / 11** | 63.6% of races |

---

## ⚙️ Feature Architecture

Features are defined in [`src/features/feature_config.py`](file:///src/features/feature_config.py) and categorized into core groups:

1. **Qualifying & Grid**: `GridPosition`, `gaptopole_bestquali`, `PitLaneStart`, `HasQualiTime`, `HasGapToPole`
2. **Teammate Comparison**: `TeammateQualifyingGap`, `HasTeammateGap`
3. **Championship Standings (Pre-Race)**: `DriverChampionshipPoints`, `DriverChampionshipPosition`, `ConstructorChampionshipPoints`, `ConstructorChampionshipPosition`
4. **Recent Form (Last 3 & 5 Races)**: `AverageFinishLast5`, `AverageFinishLast3`, `AverageGridLast3`, `ConstructorAverageFinishLast3`
5. **Circuit Performance (2023+ History)**: `StreetCircuitPerformance`, `PermanentCircuitPerformance`, `HighSpeedCircuitPerformance`, `HighDownforceCircuitPerformance` + respective history flags

---

## 📂 Repository Structure

```text
Universal-F1-Prediction-Framework/
├── data/                       # Raw datasets, feature tables, and predictions (gitignored)
├── src/
│   ├── data_collection/        # FastF1 collectors (collector.py)
│   ├── preprocessing/          # Data cleaning and type enforcement (clean.py)
│   ├── features/               # Feature modules & orchestrator (build_features.py)
│   ├── models/                 # Model factory & hyperparameter definitions (catboost_model.py)
│   ├── validation/             # Chronological walk-forward validator (walkforward.py)
│   └── prediction/             # General race prediction engine (predict.py)
├── experiments/
│   ├── baselines/              # Logistic regression, decision trees, random forests
│   ├── model_comparison/       # Ablation, stability, and tuning experiments
│   ├── eda/                    # Exploratory data analysis scripts
│   ├── historical_reference/   # Reference 60-series monolithic scripts
│   └── archive/                # Early iterative development scripts (mle1..mle59)
├── benchmark_circuit.py        # Circuit feature vectorization benchmark
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Yashash-Mathur/Universal-F1-Prediction-Framework.git
cd Universal-F1-Prediction-Framework
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Build Features
```python
import pandas as pd
from src.preprocessing.clean import clean_raw
from src.features.build_features import build_features

raw = pd.read_csv("data/f1_2020_2026_raw_v2.csv")
cleaned = clean_raw(raw)
features = build_features(cleaned)
features.to_csv("data/f1_2020_2026_features_v6.csv", index=False)
```

### 3. Run Walk-Forward Validation
```python
import pandas as pd
from src.validation.walkforward import run_walkforward

df = pd.read_csv("data/f1_2020_2026_features_v6.csv")
results, summary = run_walkforward(df, test_year=2026)
print(summary)
```

### 4. Run Inference on a Target Race
```bash
python -m src.prediction.predict \
    --features data/f1_2020_2026_features_v6.csv \
    --target data/f1_2020_2026_dutch_gp_prediction_fixed.csv \
    --train-year 2026 \
    --train-through-round 11 \
    --output data/predictions_dutch_gp.csv
```

---

## 🗺️ Roadmap

- [x] **Phase 1: ML Engine Cleanup & Parity Validation** (Modularized `src/`, 374× vectorized circuit features, repository hygiene).
- [ ] **Phase 2: PostgreSQL & REST Backend** (Relational storage for races, drivers, and precomputed predictions; FastAPI service).
- [ ] **Phase 3: Redis Rate Limiting & Containerization** (Tiered API rate limiting, token-bucket controls, Docker Compose deployment).
