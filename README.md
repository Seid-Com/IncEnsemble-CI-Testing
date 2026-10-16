# IncEnsemble: Efficient-Machine-Learning for Automated-Software-Testing

An incremental machine learning framework for automated regression test prioritization in CI/CD environments. Combines adaptive ensemble learning with online model updates to prioritize tests, detect faults, and mitigate flaky test impact.

## Key Features

- **Incremental Ensemble Learning**: Static decision forest (100 trees, depth 8) + online linear corrector (SGD with balanced class weighting)
- **13 Predictive Features**: Commit-level, code-level, historical test, change-impact, and stability features
- **Flakiness-Aware Prioritization**: Tunable penalty factor for unstable tests
- **Budget-Constrained Selection**: Optimized-test selection under execution time constraints
- **Active Learning**: Entropy/margin-based uncertainty sampling for label-efficient model updates
- **SHAP Explainability**: Feature importance analysis for transparent decision-making
- **Statistical Validation**: Wilcoxon signed-rank tests, multi-run evaluation, hyperparameter sensitivity analysis

## Architecture

```
IncEnsemble = α · StaticForest(F) + (1-α) · OnlineCorrector(L)

Score(t_i) = (P(t_i) × urgency) / cost - λ × flakiness
```

| Component | Description |
|-----------|-------------|
| Static Forest (F) | 100 shallow decision trees trained offline on historical data |
| Online Linear Corrector (L) | SGD logistic regression with `class_weight='balanced'` for incremental updates |
| Meta-weighting Layer | Convex combination: α·F + (1-α)·L (default α=0.6) |
| Flakiness Detector | Transition-based + variance scoring with configurable weights |

## Features (13 total)

| Category | Features |
|----------|----------|
| Commit-level | files_modified, lines_added, lines_deleted, code_churn |
| Code-level | token_count, method_count_delta, syntactic_complexity |
| Historical test | recent_pass_rate, avg_execution_time, failure_streak |
| Change-impact | coverage_overlap, time_since_last_failure |
| Stability | execution_time_variance |

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy pandas scikit-learn scipy plotly streamlit shap matplotlib joblib
```

## Usage

### Launch Streamlit App

```bash
streamlit run app.py
```

### Data Sources

1. **Synthetic (Demo)**: Generated data simulating 200 tests across 50 commits
2. **Defects4J Benchmark**: 17 Java projects with real bug data (Chart, Closure, Lang, Math, Time, etc.)
3. **Upload CSV**: Custom execution logs, commits, and test suites
4. **Upload XML**: JUnit/TestNG test reports

### Programmatic Usage

```python
from inc_ensemble import IncEnsemble, FlakinessDetector, TestPrioritizer
from feature_extractor import FeatureExtractor, DataSimulator

# Initialize
simulator = DataSimulator(n_tests=200, n_commits=50, failure_rate=0.15, seed=42)
test_suite = simulator.generate_test_suite()
commits = simulator.generate_commits()
test_history, execution_log = simulator.simulate_test_execution(test_suite, commits)

# Train
model = IncEnsemble(n_trees=100, max_depth=8, alpha=0.6)
# ... extract features and call model.fit_static(X, y)

# Predict and prioritize
fault_proba = model.predict_proba(X)
prioritizer = TestPrioritizer(lambda_penalty=0.3)
scores = prioritizer.compute_scores(fault_proba, execution_costs, flakiness_array)
selected, cost = prioritizer.select_tests(scores, execution_costs, budget)

# Update online
model.update_online(X_selected, y_selected)
```

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| FDR | Fault Detection Rate: fraction of failures detected within budget |
| ATD | Average Time-to-Detect: wall-clock time to first failure |
| WC | Wasted Cost: execution time spent on passing tests |
| FPR | False Positive Rate: tests predicted as failures that pass |
| ITT | Inference Time per Test: average scoring latency |
| MUL | Model Utility Loss: weighted cost of missed failures |
| Precision | Fraction of selected tests that actually fail |

### Baselines

- **Random Selection**: Tests selected uniformly at random
- **Static Prioritization**: Ranked by historical failure frequency
- **Non-incremental RF**: Standard RandomForest without online updates
- **Full Execution**: All tests run (oracle upper bound)

### Running Evaluation

```python
from baselines import (
    run_baselines, compute_fdr_vs_budget, run_multi_evaluation,
    run_hyperparameter_sensitivity, run_imbalance_comparison, wilcoxon_test
)

# Baseline comparison
results = run_baselines(system, test_commits, test_suite, budget_pct=30, n_train=35)

# FDR vs budget curve
budget_df = compute_fdr_vs_budget(system, test_suite, test_history, n_train=35)

# Multi-run evaluation (10 runs)
multi_results = run_multi_evaluation(system, n_runs=10, budget_pct=30, n_train=35)

# Hyperparameter sensitivity
sensitivity = run_hyperparameter_sensitivity(system, budget_pct=30, n_train=35)

# Imbalance strategy comparison
imbalance = run_imbalance_comparison(system, budget_pct=30, n_train=35)

# Statistical significance
test_result = wilcoxon_test(results, 'IncEnsemble', 'Random')
```

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_trees | 100 | Number of decision trees in static forest |
| max_depth | 8 | Maximum tree depth |
| alpha | 0.6 | Meta-weight for forest vs online corrector |
| learning_rate | 0.01 | SGD learning rate (adaptive) |
| lambda_penalty | 0.3 | Flakiness penalty factor |
| window_size | 20 | Flakiness calculation window |

## Project Structure

```
IncEnsemble-CI-Testing/
├── app.py                  # Streamlit web interface (13 tabs)
├── inc_ensemble.py         # Core ML models (IncEnsemble, FlakinessDetector, TestPrioritizer)
├── feature_extractor.py    # Feature extraction (13 features) + DataSimulator
├── active_learner.py       # Active learning strategies (entropy, margin, QBC)
├── baselines.py            # Baseline methods, evaluation, statistical tests
├── data_loader.py          # CSV/Defects4J data loading
├── test_report_parser.py   # JUnit/TestNG XML parsing
├── main.py                 # Entry point
├── pyproject.toml          # Project configuration
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## Streamlit App Tabs

| Tab | Description |
|-----|-------------|
| Dashboard | System overview, failure rates, execution time distribution |
| Test Prioritization | Interactive test selection with export (JSON/YAML/Shell) |
| Active Learning | Uncertainty-based test selection with exploration/exploitation |
| Flakiness Analysis | Flaky test detection and module-level breakdown |
| Performance Metrics | FDR, precision, ATD, WC, FPR over time |
| Concept Drift | Prediction error evolution, calibration, drift detection |
| Model Explainability | Feature importance, architecture description |
| SHAP Analysis | SHAP summary plots, feature contribution percentages |
| Baseline Comparison | IncEnsemble vs Random/Static/Non-inc RF with Wilcoxon tests |
| FDR vs Budget | Aggregated FDR curve with confidence bands |
| Multi-Run Evaluation | Robustness across multiple random seeds |
| Hyperparameter Sensitivity | λ and α sensitivity analysis |
| Imbalance Strategies | No balancing vs Oversampling vs Weighted learning |

## Reproducing the Paper Experiments

The `experiments/` directory contains the full scientific pipeline behind the study
(Defects4J-anchored CI-trace simulation; no network or GPU required; ~2 h on a laptop CPU):

```bash
# 1) Run all experiment families (seeds 0-9; writes results/*.csv)
python experiments/run_experiments.py

# 2) Merge remaining families (overhead, update latency, sensitivity, drift, ...)
python experiments/finish_experiments.py

# 3) Regenerate study figures from results/experiment_results.json
python experiments/make_figures.py
```

- `experiments/benchmark.py` — synthetic CI-trace generator calibrated on
  Defects4J project statistics (5 profiles, 150 bug-fix commits, 2,350 tests).
- `experiments/run_experiments.py` — main, ablation, imbalance, scalability,
  commit-FDR, and statistics experiments.
- `experiments/finish_experiments.py` — overhead, update-latency, sensitivity,
  drift, and SHAP families; merges everything into `results/experiment_results.json`.
- `experiments/make_figures.py` — study figures (FDR/ATD vs budget, sensitivity,
  adaptation, SHAP) into `results/figures/`.

All published numbers are means over 10 seeds; CSVs in `results/` are the exact
tables reported in the study.



