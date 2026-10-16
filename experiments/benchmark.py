"""Defects4J-anchored synthetic CI-trace benchmark for the IncEnsemble study.

Generates reproducible commit/test streams for five project profiles whose
aggregate scale matches the manuscript (150 bug-fix commits, 2,350 tests).
Outcomes are produced by a transparent generative model with an injected
concept-drift point, so that incremental adaptation can be evaluated under
controlled conditions.

Feature construction is strictly causal: the feature matrix for commit k is
computed only from outcomes of commits < k (sliding window W), matching the
semantics of FeatureExtractor.extract_test_features when it is fed only the
history available before the commit.
"""

import hashlib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'files_modified', 'lines_added', 'lines_deleted', 'code_churn',
    'recent_pass_rate', 'avg_execution_time', 'failure_streak',
    'coverage_overlap', 'time_since_last_failure', 'execution_time_variance',
    'token_count', 'method_count_delta', 'syntactic_complexity'
]

# name: (n_tests, n_commits, n_bug_commits)
# Totals: 2350 tests, 730 commits, 150 bug-fix commits.
PROJECT_PROFILES = {
    'Chart':   (420, 120, 25),
    'Closure': (650, 200, 40),
    'Lang':    (400, 150, 30),
    'Math':    (530, 160, 35),
    'Time':    (350, 100, 20),
}

N_MODULES = 8
WINDOW = 10
DRIFT_FRACTION = 0.6


def coverage_overlap(commit_id, test_id):
    h = hashlib.md5(f"{commit_id}:{test_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 1000) / 1000.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_project(project, seed, profile=None, drift=True):
    """Simulate one project's CI stream.

    Returns dict with:
      test_suite : DataFrame(test_id, name, module, execution_time,
                             base_failure_prob, flakiness)
      commits    : DataFrame(commit_id, timestamp, files_modified, lines_added,
                             lines_deleted, token_count, method_count_delta,
                             syntactic_complexity, is_bug)
      Y, E       : (n_commits x n_tests) outcome and execution-time matrices
      drift_index: commit index where the drift is injected (-1 if none)
    """
    if profile is None:
        profile = PROJECT_PROFILES[project]
    n_tests, n_commits, n_bug = profile
    rng = np.random.RandomState(seed)

    # ---- tests ----
    modules = rng.randint(0, N_MODULES, size=n_tests)
    execution_time = rng.lognormal(1.5, 0.8, size=n_tests)
    base_prob = rng.beta(2.0, 30.0, size=n_tests)
    flakiness = np.where(
        rng.random(n_tests) < 0.06,
        rng.beta(2.0, 12.0, size=n_tests),  # flaky minority
        rng.beta(1.0, 60.0, size=n_tests)   # stable majority
    )
    test_ids = [f'{project}_test_{i:04d}' for i in range(n_tests)]
    test_suite = pd.DataFrame({
        'test_id': test_ids,
        'name': [f'{project}::test_{i:04d}' for i in range(n_tests)],
        'module': [f'module_{m}' for m in modules],
        'execution_time': execution_time,
        'base_failure_prob': base_prob,
        'flakiness': flakiness,
    })

    # ---- commits ----
    bug_flags = np.zeros(n_commits, dtype=int)
    bug_idx = rng.choice(n_commits, size=n_bug, replace=False)
    bug_flags[bug_idx] = 1
    files_modified = np.where(bug_flags, rng.poisson(8, n_commits) + 2,
                              rng.poisson(3, n_commits) + 1)
    lines_added = np.where(bug_flags, rng.poisson(90, n_commits),
                           rng.poisson(25, n_commits))
    lines_deleted = np.where(bug_flags, rng.poisson(55, n_commits),
                             rng.poisson(15, n_commits))
    code_churn = lines_added + lines_deleted
    timestamps = pd.Timestamp('2024-01-01') + pd.to_timedelta(
        np.arange(n_commits) * 6, unit='h')
    commits = pd.DataFrame({
        'commit_id': [f'{project}_commit_{i:04d}' for i in range(n_commits)],
        'timestamp': timestamps,
        'files_modified': files_modified,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'token_count': lines_added * 8 + lines_deleted * 4,
        'method_count_delta': files_modified // 3,
        'syntactic_complexity': code_churn / np.maximum(files_modified, 1),
        'is_bug': bug_flags,
    })

    # ---- outcomes ----
    drift_index = int(n_commits * DRIFT_FRACTION) if drift else -1

    # Module-localised, rotating change impact: each commit affects a small
    # set of modules whose centre rotates over time, so (i) risk is
    # change-localised and (ii) historically buggy modules go stale.
    hot_before = rng.choice(N_MODULES, size=3, replace=False)
    hot_after = rng.choice(
        [m for m in range(N_MODULES) if m not in hot_before], size=3,
        replace=False)

    Y = np.zeros((n_commits, n_tests), dtype=np.int8)
    Y_true = np.zeros((n_commits, n_tests), dtype=np.int8)
    E = np.zeros((n_commits, n_tests), dtype=np.float64)
    cov_matrix = np.zeros((n_commits, n_tests))
    max_churn = float(code_churn.max())
    w_cov, w_churn, w_bug = 1.6, 0.5, 0.6
    logit_base = np.log(np.clip(base_prob, 1e-4, 0.5) /
                        (1 - np.clip(base_prob, 1e-4, 0.5)))
    phase = rng.uniform(0, 2 * np.pi, size=n_tests)       # per-test risk phase
    hot_phase = rng.uniform(0, 2 * np.pi, size=N_MODULES)  # per-module season
    period = max(n_commits / 2.0, 10.0)

    for k in range(n_commits):
        centre = int(k * N_MODULES / max(n_commits, 1) * 2) % N_MODULES
        n_aff = 3 if bug_flags[k] else 2
        affected = {(centre + i) % N_MODULES for i in range(n_aff)}
        if drift and k >= drift_index:
            affected = affected | set(
                hot_after[:1])          # drift re-anchors hot modules
        in_hot = np.isin(modules, list(affected)).astype(float)
        noise = np.array([coverage_overlap(commits['commit_id'].iloc[k], tid)
                          for tid in test_ids])
        cov = np.where(in_hot > 0, 0.9 + 0.1 * noise, 0.02 * noise)
        cov_matrix[k] = cov

        seasonal = 0.35 * np.sin(2 * np.pi * k / period + phase)
        module_season = 0.5 * np.sin(2 * np.pi * k / period + hot_phase[modules])
        logit = (logit_base
                 + w_cov * cov
                 + w_churn * code_churn[k] / max_churn
                 + w_bug * bug_flags[k]
                 + seasonal
                 + module_season)
        p_fail = np.clip(_sigmoid(logit), 0.0, 0.9)
        deterministic = (rng.random(n_tests) < p_fail).astype(np.int8)
        # flaky tests occasionally produce spurious failures (never masks a
        # real failure), at a moderate rate
        spurious = ((deterministic == 0) &
                    (rng.random(n_tests) < flakiness * 0.35)).astype(np.int8)
        Y_true[k] = deterministic
        Y[k] = deterministic | spurious
        time_noise = np.where(flakiness > 0.05, 0.35, 0.1)
        E[k] = np.maximum(
            execution_time * (1 + rng.normal(0, 1, n_tests) * time_noise), 0.1)

    return {
        'project': project,
        'test_suite': test_suite,
        'commits': commits,
        'Y': Y,
        'Y_true': Y_true,
        'E': E,
        'cov_matrix': cov_matrix,
        'test_ids': test_ids,
        'drift_index': drift_index,
    }


def build_test_history(project_data):
    """Full record dict compatible with FlakinessDetector / static baseline."""
    history = {tid: [] for tid in project_data['test_ids']}
    commits = project_data['commits']
    for k in range(len(commits)):
        cid = commits['commit_id'].iloc[k]
        for j, tid in enumerate(project_data['test_ids']):
            history[tid].append({
                'outcome': int(project_data['Y'][k, j]),
                'execution_time': float(project_data['E'][k, j]),
                'commit_id': cid,
            })
    return history


def commit_row_features(commit):
    files_modified = int(commit['files_modified'])
    lines_added = int(commit['lines_added'])
    lines_deleted = int(commit['lines_deleted'])
    code_churn = lines_added + lines_deleted
    return {
        'files_modified': files_modified,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'code_churn': code_churn,
        'token_count': int(commit.get('token_count', lines_added * 8 + lines_deleted * 4)),
        'method_count_delta': int(commit.get('method_count_delta', files_modified // 3)),
        'syntactic_complexity': float(commit.get(
            'syntactic_complexity', code_churn / max(files_modified, 1))),
    }


def feature_matrix_at(project_data, k, window=WINDOW):
    """Causal feature matrix (n_tests x 13) for commit k.

    Uses only outcomes from commits [k-window, k-1] (strictly before k),
    mirroring FeatureExtractor semantics on a truncated history.
    """
    Y, E = project_data['Y'], project_data['E']
    n_commits, n_tests = Y.shape
    commit = project_data['commits'].iloc[k]
    static = commit_row_features(commit)

    X = np.zeros((n_tests, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, 0] = static['files_modified']
    X[:, 1] = static['lines_added']
    X[:, 2] = static['lines_deleted']
    X[:, 3] = static['code_churn']
    X[:, 10] = static['token_count']
    X[:, 11] = static['method_count_delta']
    X[:, 12] = static['syntactic_complexity']

    lo = max(0, k - window)
    if lo < k:
        Yw = Y[lo:k]
        Ew = E[lo:k]
        X[:, 4] = 1.0 - Yw.mean(axis=0)              # recent_pass_rate
        X[:, 5] = Ew.mean(axis=0)                    # avg_execution_time
        # distance (in commits) from k-1 back to most recent failure
        X[:, 8] = np.where(Yw.any(axis=0),
                           np.argmax(Yw[::-1], axis=0), Yw.shape[0])
        X[:, 9] = Ew.var(axis=0)
    else:
        X[:, 4] = 0.5
        X[:, 5] = 1.0
        X[:, 8] = window

    streak = np.zeros(n_tests, dtype=np.float64)
    for back in range(min(k, window)):
        row = Y[k - 1 - back]
        streak = np.where((streak == back) & (row == 1), back + 1, streak)
    X[:, 6] = streak

    X[:, 7] = project_data['cov_matrix'][k]
    return X


def split_indices(project_data, train_ratio=0.7):
    n = len(project_data['commits'])
    n_train = int(n * train_ratio)
    return n_train, list(range(n_train, n))


def _self_test():
    """Validate vectorised features against FeatureExtractor on truncated history."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from feature_extractor import FeatureExtractor

    data = build_project('Lang', seed=7, profile=(40, 60, 12), drift=True)
    fe = FeatureExtractor()
    ok = True
    for k in [5, 30, 59]:
        X_fast = feature_matrix_at(data, k)
        history = {tid: [] for tid in data['test_ids']}
        for kk in range(k):
            cid = data['commits']['commit_id'].iloc[kk]
            for j, tid in enumerate(data['test_ids']):
                history[tid].append({
                    'outcome': int(data['Y'][kk, j]),
                    'execution_time': float(data['E'][kk, j]),
                    'commit_id': cid,
                })
        commit = data['commits'].iloc[k]
        commit_data = commit_row_features(commit)
        X_ref = np.array([
            fe.extract_test_features(tid, history,
                                     dict(commit_data,
                                          coverage_overlap=data['cov_matrix'][k][j]))
            for j, tid in enumerate(data['test_ids'])
        ])
        if not np.allclose(X_fast, X_ref, atol=1e-8):
            diff = np.abs(X_fast - X_ref).max(axis=0)
            print(f'MISMATCH at k={k}: max per-feature diff={diff}')
            ok = False
    print('SELF-TEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    _self_test()
