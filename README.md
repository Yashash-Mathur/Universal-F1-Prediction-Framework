# Universal F1 Prediction Framework

An end-to-end machine learning system built to predict Formula 1 race outcomes using purely pre-race data.

## 🏎️ Project Objective
The goal of this project is to build a machine learning pipeline that predicts whether a driver will finish on the **Podium (Top 3)** using features available *before* the lights go out. This project prioritizes learning the comprehensive ML lifecycle—from robust historical data collection to live model deployment—over simple hyperparameter tweaking.

## 📊 Data Evolution & Source
Data is engineered dynamically using the **FastF1 API**.
* **V1:** Historical data collection (2020–2025) featuring a robust, resumable checkpoint-saving architecture.
* **V2:** Resolved critical `GapToPole` calculation bugs by anchoring references to the absolute fastest qualifying session lap (Q1/Q2/Q3), eliminating invalid data leakage.
* **V3:** Built a finalized, clean modeling dataset removing rows with missing finishing data and applying the binary classification target: `Podium = Position <= 3`.

## ⚙️ Feature Engineering
The model leverages engineered features grouped into four core pillars:
* **Qualifying Performance:** `GridPosition`, `GapToPoleBestQuali`, `TeammateQualifyingGap`, `PitLaneStart`
* **Championship Standings:** Driver & Constructor points and current standings positions.
* **Form Metrics:** Rolling averages (`AverageFinishLast5`, `AverageGridLast3`, etc.)
* **Track Contextual Flags:** Categorized historical circuit variants (Street vs. Permanent tracks; High-Speed vs. High-Downforce layouts).

*Strict Anti-Data-Leakage Rule: All features are calculated purely using information contextually available before the specific race begins.*

## 🤖 Model & Tuning Performance
* **Baseline Model:** Logistic Regression
* **Final Classifier:** CatBoostClassifier
* **Optimized Hyperparameters:** `Depth: 6`, `Learning Rate: 0.05`, `Iterations: 300`

### Core Metrics:
| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~91.8% |
| **Precision** | ~0.70 |
| **Recall** | ~0.78 |
| **F1-Score** | ~0.737 |

## 🔮 Real-World Live Test: 2026 Barcelona Grand Prix
The pipeline was put to the test for its first live deployment during the 2026 Spanish Grand Prix weekend.

**Model Track Predictions vs Actual Results:**
1. **P1 Prediction:** Lewis Hamilton *(Actual: P1)* ✅
2. **P2 Prediction:** George Russell *(Actual: P2)* ✅
3. **P3 Prediction:** Kimi Antonelli *(Actual: Lando Norris)* ❌ 
   * *Note: Antonelli was running comfortably in P2 before suffering a mechanical engine DNF.*

## 🚀 Future Roadmap
* **Version 2:** Integrate mechanical reliability & DNF risk profiling features.
* **Version 3:** Introduce highly localized, circuit-specific historic driver indices.
* **Version 4:** Implement Practice Pace tracking (FP1, FP2, FP3 telemetry features).
* **Version 5:** Transition target from binary classification to full continuous finishing-position generation.