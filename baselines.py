import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from inc_ensemble import TestPrioritizer, FlakinessDetector


class RandomBaseline:
    def select_tests(self, n_tests, budget, execution_costs):
        indices = np.random.permutation(n_tests)
        selected = []
        total_cost = 0
        for idx in indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(idx)
                total_cost += execution_costs[idx]
            if total_cost >= budget:
                break
        return selected, total_cost


class StaticPrioritization:
    def __init__(self):
        self.failure_history = {}
    
    def update_history(self, test_id, outcome):
        if test_id not in self.failure_history:
            self.failure_history[test_id] = []
        self.failure_history[test_id].append(outcome)
    
    def _get_failure_frequency(self, test_id):
        if test_id not in self.failure_history or len(self.failure_history[test_id]) == 0:
            return 0.0
        return np.mean(self.failure_history[test_id])
    
    def select_tests(self, test_ids, execution_costs, budget):
        frequencies = np.array([self._get_failure_frequency(tid) for tid in test_ids])
        cost_efficiency = frequencies / np.maximum(execution_costs, 0.1)
        ranked_indices = np.argsort(cost_efficiency)[::-1]
        
        selected = []
        total_cost = 0
        for idx in ranked_indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(idx)
                total_cost += execution_costs[idx]
            if total_cost >= budget:
                break
        return selected, total_cost


class NonIncrementalRF:
    def __init__(self, n_trees=100, max_depth=8):
        self.model = RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def select_tests(self, fault_proba, execution_costs, budget):
        ranked_indices = np.argsort(fault_proba)[::-1]
        selected = []
        total_cost = 0
        for idx in ranked_indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(idx)
                total_cost += execution_costs[idx]
            if total_cost >= budget:
                break
        return selected, total_cost


def run_baselines(system, test_commits, test_suite, budget_pct=30, n_train=35):
    random_baseline = RandomBaseline()
    static_baseline = StaticPrioritization()
    
    nrf = NonIncrementalRF()
    
    train_commits = system['commits'].iloc[:n_train]
    
    X_train = []
    y_train = []
    fe = system['feature_extractor']
    th = system['test_history']
    
    for _, commit in train_commits.iterrows():
        commit_data = {
            'files_modified': commit['files_modified'],
            'lines_added': commit['lines_added'],
            'lines_deleted': commit['lines_deleted'],
            'coverage_overlap': 0.5,
            'token_count': commit.get('token_count', 0),
            'method_count_delta': commit.get('method_count_delta', 0),
            'syntactic_complexity': commit.get('syntactic_complexity', 0)
        }
        for test_id in test_suite['test_id']:
            features = fe.extract_test_features(test_id, th, commit_data)
            outcomes = [h['outcome'] for h in th[test_id] if h['commit_id'] == commit['commit_id']]
            if outcomes:
                X_train.append(features)
                y_train.append(outcomes[0])
            static_baseline.update_history(test_id, outcomes[0] if outcomes else 0)
    
    if len(X_train) > 0:
        nrf.fit(np.array(X_train), np.array(y_train))
    
    results = []
    execution_costs = test_suite['execution_time'].values
    n_tests = len(test_suite)
    total_budget = execution_costs.sum() * (budget_pct / 100.0)
    
    for _, commit in test_commits.iterrows():
        commit_data = {
            'files_modified': commit['files_modified'],
            'lines_added': commit['lines_added'],
            'lines_deleted': commit['lines_deleted'],
            'coverage_overlap': 0.5,
            'token_count': commit.get('token_count', 0),
            'method_count_delta': commit.get('method_count_delta', 0),
            'syntactic_complexity': commit.get('syntactic_complexity', 0)
        }
        
        X = []
        for tid in test_suite['test_id'].values:
            X.append(fe.extract_test_features(tid, th, commit_data))
        X = np.array(X)
        
        inc_proba = system['model'].predict_proba(X)
        
        actual_failures = []
        for test_id in test_suite['test_id']:
            outcomes = [h['outcome'] for h in th[test_id] if h['commit_id'] == commit['commit_id']]
            actual_failures.append(outcomes[0] if outcomes else 0)
        actual_failures = np.array(actual_failures)
        
        prioritizer = TestPrioritizer(lambda_penalty=0.3)
        flakiness_scores = system['flakiness_detector'].calculate_flakiness(th)
        flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
        
        inc_scores = prioritizer.compute_scores(inc_proba, execution_costs, flakiness_array)
        inc_selected, inc_cost = prioritizer.select_tests(inc_scores, execution_costs, total_budget)
        
        rand_selected, rand_cost = random_baseline.select_tests(n_tests, total_budget, execution_costs)
        
        static_selected, static_cost = static_baseline.select_tests(
            test_suite['test_id'].values, execution_costs, total_budget
        )
        
        nrf_proba = nrf.predict_proba(X) if nrf.is_fitted else fault_proba
        nrf_selected, nrf_cost = nrf.select_tests(nrf_proba, execution_costs, total_budget)
        
        for name, selected in [('IncEnsemble', inc_selected), ('Random', rand_selected),
                                ('Static', static_selected), ('Non-inc RF', nrf_selected)]:
            detected = sum(1 for i in selected if actual_failures[i])
            total_failures = sum(actual_failures)
            fdr = detected / max(total_failures, 1)
            precision = detected / max(len(selected), 1)
            
            selected_set = set(selected)
            false_pos = sum(1 for i in selected if not actual_failures[i])
            true_neg = sum(1 for i in range(len(actual_failures)) if i not in selected_set and not actual_failures[i])
            fpr = false_pos / max(false_pos + true_neg, 1)
            
            results.append({
                'commit_id': commit['commit_id'],
                'method': name,
                'fdr': fdr,
                'precision': precision,
                'fpr': fpr,
                'n_selected': len(selected),
                'actual_failures': total_failures
            })
        
        for test_id in test_suite['test_id']:
            outcomes = [h['outcome'] for h in th[test_id] if h['commit_id'] == commit['commit_id']]
            static_baseline.update_history(test_id, outcomes[0] if outcomes else 0)
    
    return pd.DataFrame(results)


def compute_fdr_vs_budget(system, test_suite, test_history, budget_range=None, n_train=35):
    if budget_range is None:
        budget_range = list(range(5, 101, 5))
    
    execution_costs = test_suite['execution_time'].values
    total_cost = execution_costs.sum()
    
    test_commits = system['commits'].iloc[n_train:]
    
    results = []
    
    for budget_pct in budget_range:
        budget = total_cost * (budget_pct / 100.0)
        
        fdrs = []
        for _, commit in test_commits.iterrows():
            commit_data = {
                'files_modified': commit['files_modified'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'coverage_overlap': 0.5,
                'token_count': commit.get('token_count', 0),
                'method_count_delta': commit.get('method_count_delta', 0),
                'syntactic_complexity': commit.get('syntactic_complexity', 0)
            }
            
            X = []
            for tid in test_suite['test_id'].values:
                X.append(system['feature_extractor'].extract_test_features(tid, test_history, commit_data))
            X = np.array(X)
            
            fault_proba = system['model'].predict_proba(X)
            
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            flakiness_scores = system['flakiness_detector'].calculate_flakiness(test_history)
            flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
            
            scores = prioritizer.compute_scores(fault_proba, execution_costs, flakiness_array)
            selected, _ = prioritizer.select_tests(scores, execution_costs, budget)
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            detected = sum(1 for i in selected if actual_failures[i])
            total_failures = sum(actual_failures)
            fdr = detected / max(total_failures, 1)
            fdrs.append(fdr)
        
        results.append({
            'budget_pct': budget_pct,
            'mean_fdr': np.mean(fdrs),
            'std_fdr': np.std(fdrs),
            'min_fdr': np.min(fdrs),
            'max_fdr': np.max(fdrs)
        })
    
    return pd.DataFrame(results)


def run_multi_evaluation(system, n_runs=10, budget_pct=30, n_train=35):
    test_suite = system['test_suite']
    test_history = system['test_history']
    execution_costs = test_suite['execution_time'].values
    total_budget = execution_costs.sum() * (budget_pct / 100.0)
    
    test_commits = system['commits'].iloc[n_train:]
    
    all_run_results = []
    
    for run_idx in range(n_runs):
        np.random.seed(run_idx * 42)
        
        run_fdrs = []
        run_precisions = []
        run_atds = []
        
        for _, commit in test_commits.iterrows():
            commit_data = {
                'files_modified': commit['files_modified'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'coverage_overlap': 0.5,
                'token_count': commit.get('token_count', 0),
                'method_count_delta': commit.get('method_count_delta', 0),
                'syntactic_complexity': commit.get('syntactic_complexity', 0)
            }
            
            X = []
            for tid in test_suite['test_id'].values:
                X.append(system['feature_extractor'].extract_test_features(tid, test_history, commit_data))
            X = np.array(X)
            
            fault_proba = system['model'].predict_proba(X)
            
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            fd = FlakinessDetector(window_size=20)
            flakiness_scores = fd.calculate_flakiness(test_history)
            flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
            
            scores = prioritizer.compute_scores(fault_proba, execution_costs, flakiness_array)
            selected, cost = prioritizer.select_tests(scores, execution_costs, total_budget)
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            exec_order = [idx for idx in np.argsort(scores)[::-1] if idx in set(selected)]
            m = prioritizer.calculate_metrics(selected, actual_failures, execution_costs, exec_order)
            
            run_fdrs.append(m['fault_detection_rate'])
            run_precisions.append(m['precision'])
            run_atds.append(m['atd'])
        
        all_run_results.append({
            'run': run_idx,
            'mean_fdr': np.mean(run_fdrs),
            'mean_precision': np.mean(run_precisions),
            'mean_atd': np.mean(run_atds),
            'std_fdr': np.std(run_fdrs)
        })
    
    return pd.DataFrame(all_run_results)


def run_hyperparameter_sensitivity(system, budget_pct=30, n_train=35):
    test_suite = system['test_suite']
    test_history = system['test_history']
    execution_costs = test_suite['execution_time'].values
    total_budget = execution_costs.sum() * (budget_pct / 100.0)
    
    test_commits = system['commits'].iloc[n_train:]
    
    param_results = []
    
    lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for lam in lambda_values:
        fdrs = []
        for _, commit in test_commits.iterrows():
            commit_data = {
                'files_modified': commit['files_modified'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'coverage_overlap': 0.5,
                'token_count': commit.get('token_count', 0),
                'method_count_delta': commit.get('method_count_delta', 0),
                'syntactic_complexity': commit.get('syntactic_complexity', 0)
            }
            
            X = []
            for tid in test_suite['test_id'].values:
                X.append(system['feature_extractor'].extract_test_features(tid, test_history, commit_data))
            X = np.array(X)
            
            fault_proba = system['model'].predict_proba(X)
            
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            fd = FlakinessDetector(window_size=20)
            flakiness_scores = fd.calculate_flakiness(test_history)
            flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
            
            scores = prioritizer.compute_scores(fault_proba, execution_costs, flakiness_array)
            selected, _ = prioritizer.select_tests(scores, execution_costs, total_budget)
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            detected = sum(1 for i in selected if actual_failures[i])
            total_failures = sum(actual_failures)
            fdr = detected / max(total_failures, 1)
            fdrs.append(fdr)
        
        param_results.append({
            'parameter': 'lambda',
            'value': lam,
            'mean_fdr': np.mean(fdrs),
            'std_fdr': np.std(fdrs)
        })
    
    for alp in alpha_values:
        original_alpha = system['model'].alpha
        system['model'].alpha = alp
        
        fdrs = []
        for _, commit in test_commits.iterrows():
            commit_data = {
                'files_modified': commit['files_modified'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'coverage_overlap': 0.5,
                'token_count': commit.get('token_count', 0),
                'method_count_delta': commit.get('method_count_delta', 0),
                'syntactic_complexity': commit.get('syntactic_complexity', 0)
            }
            
            X = []
            for tid in test_suite['test_id'].values:
                X.append(system['feature_extractor'].extract_test_features(tid, test_history, commit_data))
            X = np.array(X)
            
            fault_proba = system['model'].predict_proba(X)
            
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            fd = FlakinessDetector(window_size=20)
            flakiness_scores = fd.calculate_flakiness(test_history)
            flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
            
            scores = prioritizer.compute_scores(fault_proba, execution_costs, flakiness_array)
            selected, _ = prioritizer.select_tests(scores, execution_costs, total_budget)
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            detected = sum(1 for i in selected if actual_failures[i])
            total_failures = sum(actual_failures)
            fdr = detected / max(total_failures, 1)
            fdrs.append(fdr)
        
        param_results.append({
            'parameter': 'alpha',
            'value': alp,
            'mean_fdr': np.mean(fdrs),
            'std_fdr': np.std(fdrs)
        })
        
        system['model'].alpha = original_alpha
    
    return pd.DataFrame(param_results)


def wilcoxon_test(results_df, method_a='IncEnsemble', method_b='Random'):
    a_fdrs = results_df[results_df['method'] == method_a]['fdr'].values
    b_fdrs = results_df[results_df['method'] == method_b]['fdr'].values
    
    min_len = min(len(a_fdrs), len(b_fdrs))
    a_fdrs = a_fdrs[:min_len]
    b_fdrs = b_fdrs[:min_len]
    
    if min_len < 5:
        return {'test': 'wilcoxon', 'method_a': method_a, 'method_b': method_b,
                'statistic': 0.0, 'p_value': 1.0, 'significant': False,
                'message': 'Insufficient data for statistical test'}
    
    diff = a_fdrs - b_fdrs
    
    if np.all(diff == 0):
        return {'test': 'wilcoxon', 'method_a': method_a, 'method_b': method_b,
                'statistic': 0.0, 'p_value': 1.0, 'significant': False,
                'message': 'No difference between methods'}
    
    try:
        statistic, p_value = stats.wilcoxon(diff, alternative='two-sided')
    except ValueError:
        statistic, p_value = 0.0, 1.0
    
    return {
        'test': 'wilcoxon',
        'method_a': method_a,
        'method_b': method_b,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'message': f"{'Significant' if p_value < 0.05 else 'Not significant'} difference (p={p_value:.4f})"
    }


def run_imbalance_comparison(system, budget_pct=30, n_train=35):
    from sklearn.utils import resample
    
    test_suite = system['test_suite']
    test_history = system['test_history']
    execution_costs = test_suite['execution_time'].values
    total_budget = execution_costs.sum() * (budget_pct / 100.0)
    
    train_commits = system['commits'].iloc[:n_train]
    test_commits = system['commits'].iloc[n_train:]
    
    fe = system['feature_extractor']
    
    X_all = []
    y_all = []
    for _, commit in train_commits.iterrows():
        commit_data = {
            'files_modified': commit['files_modified'],
            'lines_added': commit['lines_added'],
            'lines_deleted': commit['lines_deleted'],
            'coverage_overlap': 0.5,
            'token_count': commit.get('token_count', 0),
            'method_count_delta': commit.get('method_count_delta', 0),
            'syntactic_complexity': commit.get('syntactic_complexity', 0)
        }
        for test_id in test_suite['test_id']:
            features = fe.extract_test_features(test_id, test_history, commit_data)
            outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
            if outcomes:
                X_all.append(features)
                y_all.append(outcomes[0])
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    strategies = {}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    rf_no_bal = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_no_bal.fit(X_scaled, y_all)
    strategies['No balancing'] = rf_no_bal
    
    minority_idx = np.where(y_all == 1)[0]
    majority_idx = np.where(y_all == 0)[0]
    
    if len(minority_idx) > 0 and len(majority_idx) > 0:
        minority_upsampled = resample(minority_idx, replace=True, n_samples=len(majority_idx), random_state=42)
        balanced_idx = np.concatenate([majority_idx, minority_upsampled])
        np.random.shuffle(balanced_idx)
        
        rf_oversample = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf_oversample.fit(scaler.transform(X_all[balanced_idx]), y_all[balanced_idx])
        strategies['Random Oversampling'] = rf_oversample
    
    rf_balanced = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf_balanced.fit(X_scaled, y_all)
    strategies['Weighted Learning'] = rf_balanced
    
    results = []
    
    for name, model in strategies.items():
        fdrs = []
        precisions = []
        
        for _, commit in test_commits.iterrows():
            commit_data = {
                'files_modified': commit['files_modified'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'coverage_overlap': 0.5,
                'token_count': commit.get('token_count', 0),
                'method_count_delta': commit.get('method_count_delta', 0),
                'syntactic_complexity': commit.get('syntactic_complexity', 0)
            }
            
            X = []
            for tid in test_suite['test_id'].values:
                X.append(fe.extract_test_features(tid, test_history, commit_data))
            X = np.array(X)
            
            proba = model.predict_proba(scaler.transform(X))[:, 1]
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in test_history[test_id] if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            scores = prioritizer.compute_scores(proba, execution_costs, np.zeros(len(proba)))
            selected, _ = prioritizer.select_tests(scores, execution_costs, total_budget)
            
            detected = sum(1 for i in selected if actual_failures[i])
            total_failures = sum(actual_failures)
            fdr = detected / max(total_failures, 1)
            precision = detected / max(len(selected), 1)
            
            fdrs.append(fdr)
            precisions.append(precision)
        
        recall = np.mean(fdrs)
        f1 = 2 * np.mean(precisions) * recall / max(np.mean(precisions) + recall, 1e-10)
        
        results.append({
            'method': name,
            'f1': f1,
            'recall': recall,
            'precision': np.mean(precisions)
        })
    
    return pd.DataFrame(results)
