"""Paper experiments for IncEnsemble (IET Software revision).

Produces results/experiment_results.json (+ CSVs) used for every table/figure of
the revised manuscript:
  1. Main comparison  : 6 methods x 3 budgets x 5 projects x 10 seeds
  2. Statistics       : mean, SD, 95% CI, paired Wilcoxon, Cliff's delta
  3. Ablation         : component removal + measured overhead
  4. Sensitivity      : lambda, alpha, learning rate, depth, trees
  5. Imbalance        : no-balancing / oversampling / SMOTE / weighted SGD
  6. Scalability      : 500-5000 tests (ITT, update latency)
  7. Drift            : pre/post-drift FDR (IncEnsemble vs NonIncRF)
  8. Update latency   : online update vs full retraining per commit
  9. SHAP             : mean |SHAP| per feature

Run:  python experiments/run_experiments.py
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learner import ActiveLearner
from experiments.benchmark import (PROJECT_PROFILES, build_project,
                                         build_test_history, feature_matrix_at,
                                         split_indices, FEATURE_NAMES)
from inc_ensemble import IncEnsemble, FlakinessDetector, TestPrioritizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
SEEDS = list(range(10))
BUDGETS = [10, 25, 50]
METHODS = ['IncEnsemble', 'Random', 'Static', 'NonIncRF', 'XGBoost', 'L2R']
LAMBDA_PENALTY = 0.3
ALPHA = 0.6
EXPLORATION_RATIO = 0.1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class PaperFlakinessDetector:
    """Flakiness = unstable runtime (CV) + outcome intermittency in excess of
    what the observed base rate would produce. Runtime instability is the key
    observable handle: flaky tests exhibit erratic execution times."""

    def __init__(self, window_size=20):
        self.window_size = window_size

    def calculate_flakiness(self, test_history):
        scores = {}
        for tid, records in test_history.items():
            if len(records) < 3:
                scores[tid] = 0.0
                continue
            recent = records[-self.window_size:]
            times = np.array([r['execution_time'] for r in recent], dtype=float)
            outcomes = np.array([r['outcome'] for r in recent], dtype=float)
            cv = times.std() / max(times.mean(), 1e-9)
            cv_component = min(cv / 0.5, 1.0)
            trans = np.mean(outcomes[1:] != outcomes[:-1])
            p = outcomes.mean()
            expected = 2 * p * (1 - p)
            excess = max(0.0, trans - expected)
            scores[tid] = float(min(0.6 * cv_component + 0.4 * min(excess / 0.5, 1.0), 1.0))
        return scores


def greedy_select(scores, costs, budget):
    order = np.argsort(scores)[::-1]
    selected, total = [], 0.0
    for idx in order:
        if total + costs[idx] <= budget:
            selected.append(int(idx))
            total += costs[idx]
        if total >= budget:
            break
    return selected, total


def random_select(n_tests, costs, budget, rng):
    order = rng.permutation(n_tests)
    selected, total = [], 0.0
    for idx in order:
        if total + costs[idx] <= budget:
            selected.append(int(idx))
            total += costs[idx]
        if total >= budget:
            break
    return selected, total


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    more = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((more - less) / (len(a) * len(b)))


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return [float(values.mean()), float(values.mean())]
    se = values.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return [float(values.mean() - tcrit * se), float(values.mean() + tcrit * se)]


def manual_smote(X, y, random_state=42, k=5):
    rng = np.random.RandomState(random_state)
    minority = np.where(y == 1)[0]
    majority = np.where(y == 0)[0]
    if len(minority) == 0 or len(majority) <= len(minority):
        return X, y
    Xm = X[minority]
    n_gen = len(majority) - len(minority)
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k, len(Xm) - 1)
    nn_model = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(Xm)
    _, idxs = nn_model.kneighbors(Xm)
    nn = idxs[:, 1:]
    synth = []
    for i in range(n_gen):
        src = rng.randint(0, len(Xm))
        nb = nn[src][rng.randint(0, k_eff)]
        gap = rng.random()
        synth.append(Xm[src] + gap * (Xm[nb] - Xm[src]))
    X_new = np.vstack([X, np.array(synth)])
    y_new = np.concatenate([y, np.ones(n_gen, dtype=y.dtype)])
    return X_new, y_new


# --------------------------------------------------------------------------- #
# experiment 1: main comparison
# --------------------------------------------------------------------------- #
def fit_static_models(X_train, y_train, seed, with_online_sgd=False):
    """Fit the non-incremental baselines once per (project, seed)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                random_state=42, n_jobs=-1)
    rf.fit(Xs, y_train)

    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb.fit(Xs, y_train)

    l2r = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    l2r.fit(Xs, y_train)

    return {'scaler': scaler, 'rf': rf, 'xgb': xgb, 'l2r': l2r,
            'y_mean': float(np.mean(y_train))}


def build_train_matrix(project_data, n_train, window=10):
    X_rows, y_rows = [], []
    for k in range(n_train):
        X_rows.append(feature_matrix_at(project_data, k, window))
        y_rows.append(project_data['Y'][k])
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    return X, y


def run_single(project_data, budget_pct, seed, config=None, collect_commit=False):
    """Run all methods on one project at one budget. Returns per-method metrics."""
    cfg = {'alpha': ALPHA, 'lam': LAMBDA_PENALTY, 'active': True,
           'online': True, 'lr': 0.01, 'depth': 8, 'trees': 100}
    if config:
        cfg.update(config)

    n_train, test_idx = split_indices(project_data)
    test_suite = project_data['test_suite']
    costs = test_suite['execution_time'].values.astype(float)
    n_tests = len(test_suite)
    total_budget = costs.sum() * budget_pct / 100.0

    X_train, y_train = build_train_matrix(project_data, n_train)
    static_models = fit_static_models(X_train, y_train, seed)

    model = IncEnsemble(n_trees=cfg['trees'], max_depth=cfg['depth'],
                        alpha=cfg['alpha'])
    model.online_corrector.set_params(eta0=cfg['lr'])
    model.fit_static(X_train, y_train)

    flaky = PaperFlakinessDetector(window_size=20)
    prioritizer = TestPrioritizer(lambda_penalty=cfg['lam'])
    learner = ActiveLearner(utility_weight=1.0, cost_weight=1.0)

    # Partial-feedback world: only executed tests produce observable outcomes.
    history = {tid: [] for tid in project_data['test_ids']}
    static_freq = {tid: [] for tid in project_data['test_ids']}
    rng = np.random.RandomState(seed * 4242)

    scaler = static_models['scaler']
    results = {m: {k: [] for k in
                   ['fdr', 'precision', 'fpr', 'atd', 'wasted_cost', 'mul']}
               for m in METHODS}
    commit_rows = []

    for k in test_idx:
        X = feature_matrix_at(project_data, k)
        actual = project_data['Y'][k].astype(int)          # observed (noisy)
        genuine = project_data.get('Y_true', project_data['Y'])[k].astype(int)
        flakiness_scores = flaky.calculate_flakiness(history)
        f_arr = np.array([flakiness_scores[t] for t in project_data['test_ids']])

        # ---- IncEnsemble (+ optional active-learning exploration) ----
        inc_proba = model.predict_proba(X)
        inc_scores = prioritizer.compute_scores(inc_proba, costs, f_arr)

        if cfg['active'] and cfg['online']:
            exp_budget = total_budget * EXPLORATION_RATIO
            entropy = learner.calculate_uncertainty(inc_proba)
            exp_utility = (entropy * np.maximum(inc_proba, 1e-6)) / \
                np.maximum(costs, 0.1)
            exp_sel, exp_cost = greedy_select(exp_utility, costs, exp_budget)
            remain_mask = np.ones(n_tests, dtype=bool)
            remain_mask[exp_sel] = False
            rem_scores = np.where(remain_mask, inc_scores, -np.inf)
            rem_budget = max(0.0, total_budget - exp_cost)
            rem_sel, _ = greedy_select(rem_scores, costs, rem_budget)
            inc_sel = list(exp_sel) + rem_sel
            inc_sel = sorted(set(inc_sel),
                             key=lambda i: inc_scores[i], reverse=True)
        else:
            inc_sel, _ = greedy_select(inc_scores, costs, total_budget)

        # ---- baselines ----
        rand_sel, _ = random_select(n_tests, costs, total_budget, rng)

        freq = np.array([
            (np.mean(static_freq[t]) if static_freq[t] else 0.0)
            for t in project_data['test_ids']])
        static_scores = freq / np.maximum(costs, 0.1)
        static_sel, _ = greedy_select(static_scores, costs, total_budget)

        Xs = scaler.transform(X)
        selections = {'IncEnsemble': inc_sel, 'Random': rand_sel,
                      'Static': static_sel,
                      'NonIncRF': greedy_select(
                          static_models['rf'].predict_proba(Xs)[:, 1],
                          costs, total_budget)[0],
                      'XGBoost': greedy_select(
                          static_models['xgb'].predict_proba(Xs)[:, 1],
                          costs, total_budget)[0],
                      'L2R': greedy_select(
                          static_models['l2r'].predict(Xs), costs,
                          total_budget)[0]}

        for name, sel in selections.items():
            m = prioritizer.calculate_metrics(sel, genuine, costs,
                                              execution_order=sel)
            results[name]['fdr'].append(m['fault_detection_rate'])
            results[name]['precision'].append(m['precision'])
            results[name]['fpr'].append(m['fpr'])
            results[name]['atd'].append(m['atd'])
            results[name]['wasted_cost'].append(m['wasted_cost'])
            results[name]['mul'].append(m['mul'])

        if collect_commit:
            commit_rows.append({
                'commit': k,
                'drift': int(k >= project_data['drift_index'])
                if project_data['drift_index'] >= 0 else 0,
                **{f'fdr_{name}': results[name]['fdr'][-1] for name in METHODS},
            })

        # ---- feedback updates (only executed tests are observed) ----
        for j in inc_sel:
            history[project_data['test_ids'][j]].append({
                'outcome': int(actual[j]),
                'execution_time': float(project_data['E'][k, j]),
                'commit_id': project_data['commits']['commit_id'].iloc[k],
            })
            static_freq[project_data['test_ids'][j]].append(int(actual[j]))
        if cfg['online'] and inc_sel:
            obs = np.array(inc_sel, dtype=int)
            model.update_online(X[obs], actual[obs])

    summary = {}
    for name in METHODS:
        summary[name] = {metric: float(np.mean(values))
                         for metric, values in results[name].items()}
    return summary, commit_rows


# --------------------------------------------------------------------------- #
# experiment 2: ablation + overhead
# --------------------------------------------------------------------------- #
ABLATION_CONFIGS = {
    'Full IncEnsemble': {},
    'Without online corrector': {'online': False},
    'Without active learning': {'active': False},
    'Without flakiness penalty': {'lam': 0.0},
    'Forest only (no updates)': {'online': False, 'active': False, 'alpha': 1.0},
}


def run_ablation(projects=('Lang', 'Math'), seeds=(0, 1, 2, 3, 4), budget_pct=10):
    rows = []
    for project in projects:
        for seed in seeds:
            data = build_project(project, seed=seed)
            for label, cfg in ABLATION_CONFIGS.items():
                summary, _ = run_single(data, budget_pct, seed, config=cfg)
                rows.append({
                    'project': project, 'seed': seed, 'config': label,
                    'fdr': summary['IncEnsemble']['fdr'],
                    'wasted_cost': summary['IncEnsemble']['wasted_cost'],
                    'atd': summary['IncEnsemble']['atd'],
                })
    return pd.DataFrame(rows)


def measure_overhead(project='Lang', seed=0):
    data = build_project(project, seed=seed)
    n_train, test_idx = split_indices(data)
    X_train, y_train = build_train_matrix(data, n_train)
    model = IncEnsemble(n_trees=100, max_depth=8, alpha=ALPHA)
    model.fit_static(X_train, y_train)
    history = build_test_history(data)
    flaky = FlakinessDetector(window_size=20)
    costs = data['test_suite']['execution_time'].values.astype(float)

    # feature extraction
    t0 = time.perf_counter()
    for k in test_idx[:10]:
        feature_matrix_at(data, k)
    feat_s = (time.perf_counter() - t0) / 10

    X = feature_matrix_at(data, test_idx[0])
    Xs = model.scaler.transform(X)
    t0 = time.perf_counter()
    for _ in range(50):
        model.static_forest.predict_proba(Xs)[:, 1]
    forest_ms = (time.perf_counter() - t0) / 50 / len(X) * 1000

    if model.online_fitted:
        model.update_online(X[:10], data['Y'][test_idx[0]][:10])
        t0 = time.perf_counter()
        for _ in range(20):
            model.update_online(X[:10], data['Y'][test_idx[0]][:10])
        update_ms = (time.perf_counter() - t0) / 20 * 1000
    else:
        model.update_online(X[:20], data['Y'][test_idx[0]][:20])
        t0 = time.perf_counter()
        for _ in range(20):
            model.update_online(X[:20], data['Y'][test_idx[0]][:20])
        update_ms = (time.perf_counter() - t0) / 20 * 1000

    t0 = time.perf_counter()
    for _ in range(20):
        flaky.calculate_flakiness(history)
    flaky_ms = (time.perf_counter() - t0) / 20 * 1000

    model_bytes = len(pickle.dumps(model.static_forest)) + len(
        pickle.dumps(model.online_corrector))

    # retraining cost of non-incremental baselines on all data so far
    X_all = np.vstack([X_train, X])
    y_all = np.concatenate([y_train, data['Y'][test_idx[0]]])
    t0 = time.perf_counter()
    RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                           random_state=42).fit(StandardScaler().fit_transform(X_all), y_all)
    rf_retrain_ms = (time.perf_counter() - t0) * 1000

    return {
        'feature_extraction_ms_per_commit': feat_s * 1000,
        'feature_extraction_ms_per_test': feat_s * 1000 / len(X),
        'forest_scoring_ms_per_test': forest_ms,
        'online_update_ms_per_commit': update_ms,
        'flakiness_ms_per_suite': flaky_ms,
        'model_size_mb': model_bytes / 1e6,
        'rf_full_retrain_ms': rf_retrain_ms,
    }


def run_update_latency(project='Lang', seed=0):
    """Per-commit: IncEnsemble update time vs full retrain time of RF."""
    data = build_project(project, seed=seed)
    n_train, test_idx = split_indices(data)
    X_train, y_train = build_train_matrix(data, n_train)
    model = IncEnsemble(n_trees=100, max_depth=8, alpha=ALPHA)
    model.fit_static(X_train, y_train)
    costs = data['test_suite']['execution_time'].values.astype(float)
    total_budget = costs.sum() * 0.25
    prioritizer = TestPrioritizer(lambda_penalty=LAMBDA_PENALTY)
    flaky = FlakinessDetector(window_size=20)
    history = build_test_history(data)
    scaler = StandardScaler().fit(X_train)

    rows = []
    X_seen, y_seen = list(X_train), list(y_train)
    for step, k in enumerate(test_idx):
        X = feature_matrix_at(data, k)
        actual = data['Y'][k].astype(int)
        scores = prioritizer.compute_scores(
            model.predict_proba(X), costs,
            np.array([flaky.calculate_flakiness(history)[t]
                      for t in data['test_ids']]))
        sel, _ = greedy_select(scores, costs, total_budget)
        t0 = time.perf_counter()
        if sel:
            model.update_online(X[sel], actual[sel])
        inc_ms = (time.perf_counter() - t0) * 1000
        X_seen.append(X[sel] if len(sel) else X[:1])
        y_seen.extend(np.atleast_1d(actual[sel] if len(sel) else actual[:1]).tolist())
        if step % 5 == 0:
            Xa = scaler.transform(np.vstack(X_seen))
            ya = np.asarray(y_seen, dtype=float)
            t0 = time.perf_counter()
            RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1,
                                   random_state=42).fit(Xa, ya)
            retrain_ms = (time.perf_counter() - t0) * 1000
        else:
            retrain_ms = np.nan
        rows.append({'commit': k, 'inc_update_ms': inc_ms,
                     'rf_retrain_ms': retrain_ms})
    df = pd.DataFrame(rows)
    df['rf_retrain_ms'] = df['rf_retrain_ms'].interpolate(limit_direction='both')
    return df


# --------------------------------------------------------------------------- #
# experiment 3: hyperparameter sensitivity
# --------------------------------------------------------------------------- #
def run_sensitivity(project='Lang', seeds=(0, 1, 2), budget_pct=25):
    grid = {
        'lambda': ([{'lam': v} for v in [0.1, 0.2, 0.3, 0.4, 0.5]], [0.1, 0.2, 0.3, 0.4, 0.5]),
        'alpha': ([{'alpha': v} for v in [0.3, 0.4, 0.5, 0.6, 0.7]], [0.3, 0.4, 0.5, 0.6, 0.7]),
        'learning_rate': ([{'lr': v} for v in [0.001, 0.005, 0.01, 0.05, 0.1]],
                          [0.001, 0.005, 0.01, 0.05, 0.1]),
        'depth': ([{'depth': v} for v in [4, 6, 8, 10]], [4, 6, 8, 10]),
        'trees': ([{'trees': v} for v in [50, 100, 150, 200]], [50, 100, 150, 200]),
    }
    rows = []
    cache = {}
    for seed in seeds:
        data = build_project(project, seed=seed)
        for param, (configs, values) in grid.items():
            for cfg, value in zip(configs, values):
                key = (seed, json.dumps(cfg, sort_keys=True))
                if key not in cache:
                    summary, _ = run_single(data, budget_pct, seed, config=cfg)
                    cache[key] = summary['IncEnsemble']
                rows.append({'parameter': param, 'value': value, 'seed': seed,
                             'fdr': cache[key]['fdr'],
                             'wasted_cost': cache[key]['wasted_cost']})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# experiment 4: imbalance strategies
# --------------------------------------------------------------------------- #
def run_imbalance(project='Lang', seeds=(0, 1, 2)):
    rows = []
    for seed in seeds:
        data = build_project(project, seed=seed)
        n_train, test_idx = split_indices(data)
        X_train, y_train = build_train_matrix(data, n_train)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)

        strategies = {}
        strategies['No balancing'] = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(Xs, y_train)
        minority = np.where(y_train == 1)[0]
        majority = np.where(y_train == 0)[0]
        up = np.random.RandomState(42).choice(
            minority, size=len(majority), replace=True)
        bal_idx = np.concatenate([majority, up])
        np.random.RandomState(42).shuffle(bal_idx)
        strategies['Random Oversampling'] = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ).fit(Xs[bal_idx], y_train[bal_idx])
        Xm, ym = manual_smote(Xs, y_train, random_state=42)
        strategies['SMOTE'] = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(Xm, ym)
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01,
                            random_state=42)
        n = len(y_train)
        n1 = max(int((y_train == 1).sum()), 1)
        n0 = max(int((y_train == 0).sum()), 1)
        sw = np.where(y_train == 1, n / (2.0 * n1), n / (2.0 * n0))
        sgd.partial_fit(Xs, y_train, classes=[0, 1], sample_weight=sw)
        strategies['Weighted online learning (proposed)'] = sgd

        X_eval = np.vstack([feature_matrix_at(data, k) for k in test_idx])
        y_eval = np.concatenate([data['Y'][k] for k in test_idx])
        for name, model in strategies.items():
            proba = model.predict_proba(scaler.transform(X_eval))[:, 1] \
                if not isinstance(model, SGDClassifier) \
                else model.predict_proba(X_eval)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                'seed': seed, 'method': name,
                'f1': f1_score(y_eval, pred, zero_division=0),
                'recall': recall_score(y_eval, pred, zero_division=0),
                'precision': precision_score(y_eval, pred, zero_division=0),
                'auc': roc_auc_score(y_eval, proba),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# experiment 5: scalability
# --------------------------------------------------------------------------- #
def run_scalability(sizes=(500, 1000, 2000, 5000), seed=0):
    rows = []
    for n_tests in sizes:
        data = build_project('Scale', seed=seed,
                             profile=(n_tests, 120, 25), drift=False)
        n_train, test_idx = split_indices(data)
        X_train, y_train = build_train_matrix(data, n_train)
        model = IncEnsemble(n_trees=100, max_depth=8, alpha=ALPHA)
        model.fit_static(X_train, y_train)
        costs = data['test_suite']['execution_time'].values.astype(float)

        t0 = time.perf_counter()
        X0 = feature_matrix_at(data, test_idx[0])
        feat_ms = (time.perf_counter() - t0) * 1000

        Xs = model.scaler.transform(X0)
        t0 = time.perf_counter()
        for _ in range(20):
            model.predict_proba(X0)
        score_ms = (time.perf_counter() - t0) / 20 * 1000
        itt = score_ms / n_tests

        model.update_online(X0[:50], data['Y'][test_idx[0]][:50])
        t0 = time.perf_counter()
        for _ in range(20):
            model.update_online(X0[:50], data['Y'][test_idx[0]][:50])
        update_ms = (time.perf_counter() - t0) / 20 * 1000

        rows.append({'n_tests': n_tests, 'feature_ms_per_commit': feat_ms,
                     'scoring_ms_per_commit': score_ms,
                     'inference_ms_per_test': itt,
                     'update_ms_per_commit': update_ms})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# experiment 6: SHAP
# --------------------------------------------------------------------------- #
def run_shap(project='Lang', seed=0, max_rows=20000):
    import shap
    data = build_project(project, seed=seed)
    n_train, _ = split_indices(data)
    X_train, y_train = build_train_matrix(data, n_train)
    model = IncEnsemble(n_trees=100, max_depth=8, alpha=ALPHA)
    model.fit_static(X_train, y_train)
    Xs = model.scaler.transform(X_train)
    if len(Xs) > max_rows:
        idx = np.random.RandomState(0).choice(len(Xs), max_rows, replace=False)
        X_sample = Xs[idx]
    else:
        X_sample = Xs
    explainer = shap.TreeExplainer(model.static_forest)
    shap_values = explainer.shap_values(X_sample)
    # normalize shape to (n_samples, n_features)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # (classes, n, f) or (n, f, classes)
        if sv.shape[0] == 2:
            sv = sv[1]
        elif sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        else:
            sv = sv.mean(axis=-1) if sv.shape[-1] != sv.shape[1] else sv.mean(axis=0)
    elif isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
    mean_abs = np.abs(sv).mean(axis=0)
    total = mean_abs.sum()
    df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'mean_abs_shap': mean_abs,
        'share_pct': 100.0 * mean_abs / total,
    }).sort_values('mean_abs_shap', ascending=False)
    return df


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def compute_statistics(main_df):
    """main_df: rows of (project, seed, budget, method, metrics...)."""
    out = {}
    for budget in BUDGETS:
        sub = main_df[main_df['budget'] == budget]
        units = sorted(set(zip(sub['project'], sub['seed'])))
        entry = {'per_method': {}, 'comparisons': {}}
        for method in METHODS:
            mrows = sub[sub['method'] == method]
            stats_out = {}
            for metric in ['fdr', 'precision', 'fpr', 'atd', 'wasted_cost', 'mul']:
                vals = mrows.set_index(['project', 'seed']).loc[units][metric].values
                stats_out[metric] = {
                    'mean': float(vals.mean()),
                    'sd': float(vals.std(ddof=1)),
                    'ci95': ci95(vals),
                }
            entry['per_method'][method] = stats_out

        def unit_values(method, metric):
            mrows = sub[sub['method'] == method]
            return mrows.set_index(['project', 'seed']).loc[units][metric].values

        for baseline in ['Random', 'Static', 'NonIncRF', 'XGBoost', 'L2R']:
            cmp_out = {}
            for metric in ['fdr', 'wasted_cost', 'atd']:
                a = unit_values('IncEnsemble', metric)
                b = unit_values(baseline, metric)
                diff = a - b
                if metric == 'wasted_cost':
                    # lower is better for baseline comparison direction
                    pass
                if np.allclose(diff, 0):
                    p, w = 1.0, 0.0
                else:
                    try:
                        w, p = stats.wilcoxon(a, b, alternative='two-sided')
                    except ValueError:
                        w, p = 0.0, 1.0
                cmp_out[metric] = {
                    'mean_diff': float(diff.mean()),
                    'ci95_diff': ci95(diff),
                    'wilcoxon_W': float(w),
                    'p_value': float(p),
                    'cliffs_delta': cliffs_delta(a, b),
                    'significant': bool(p < 0.05),
                }
            entry['comparisons'][baseline] = cmp_out
        out[str(budget)] = entry
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_main_experiment():
    rows, commit_rows_all = [], []
    for project in PROJECT_PROFILES:
        for seed in SEEDS:
            data = build_project(project, seed=seed)
            for budget in BUDGETS:
                summary, commit_rows = run_single(
                    data, budget, seed,
                    collect_commit=(project == 'Lang' and budget == 25))
                for method in METHODS:
                    rows.append({'project': project, 'seed': seed,
                                 'budget': budget, 'method': method,
                                 **summary[method]})
                if commit_rows:
                    for r in commit_rows:
                        r.update({'project': project, 'seed': seed,
                                  'budget': budget})
                        commit_rows_all.append(r)
            print(f'  main: {project} seed={seed} done', flush=True)
    main_df = pd.DataFrame(rows)
    commits_df = pd.DataFrame(commit_rows_all) if commit_rows_all else pd.DataFrame()
    return main_df, commits_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    print('== Main comparison ==', flush=True)
    main_df, commits_df = run_main_experiment()
    main_df.to_csv(os.path.join(RESULTS_DIR, 'main.csv'), index=False)
    results['main'] = main_df.to_dict(orient='records')
    results['statistics'] = compute_statistics(main_df)

    print('== Ablation ==', flush=True)
    ablation_df = run_ablation()
    ablation_df.to_csv(os.path.join(RESULTS_DIR, 'ablation.csv'), index=False)
    results['ablation'] = ablation_df.to_dict(orient='records')

    print('== Overhead ==', flush=True)
    results['overhead'] = measure_overhead()
    print('== Update latency ==', flush=True)
    upd_df = run_update_latency()
    upd_df.to_csv(os.path.join(RESULTS_DIR, 'update_latency.csv'), index=False)
    results['update_latency'] = upd_df.to_dict(orient='records')

    print('== Sensitivity ==', flush=True)
    sens_df = run_sensitivity()
    sens_df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity.csv'), index=False)
    results['sensitivity'] = sens_df.to_dict(orient='records')

    print('== Imbalance ==', flush=True)
    imb_df = run_imbalance()
    imb_df.to_csv(os.path.join(RESULTS_DIR, 'imbalance.csv'), index=False)
    results['imbalance'] = imb_df.to_dict(orient='records')

    print('== Scalability ==', flush=True)
    scale_df = run_scalability()
    scale_df.to_csv(os.path.join(RESULTS_DIR, 'scalability.csv'), index=False)
    results['scalability'] = scale_df.to_dict(orient='records')

    print('== Drift ==', flush=True)
    if not commits_df.empty:
        drift = (commits_df.groupby(['method', 'drift'])['fdr']
                 .mean().reset_index())
        results['drift'] = drift.to_dict(orient='records')
        drift.to_csv(os.path.join(RESULTS_DIR, 'drift.csv'), index=False)
        commits_df.to_csv(os.path.join(RESULTS_DIR, 'commit_fdr.csv'), index=False)

    print('== SHAP ==', flush=True)
    shap_df = run_shap()
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap.csv'), index=False)
    results['shap'] = shap_df.to_dict(orient='records')

    with open(os.path.join(RESULTS_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('Saved results/experiment_results.json', flush=True)


if __name__ == '__main__':
    main()
