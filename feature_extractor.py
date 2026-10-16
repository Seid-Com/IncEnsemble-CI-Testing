import numpy as np
import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'files_modified',
            'lines_added',
            'lines_deleted',
            'code_churn',
            'recent_pass_rate',
            'avg_execution_time',
            'failure_streak',
            'coverage_overlap',
            'time_since_last_failure',
            'execution_time_variance',
            'token_count',
            'method_count_delta',
            'syntactic_complexity'
        ]
    
    def extract_test_features(self, test_id, test_history, commit_data, window_size=10):
        if test_id not in test_history or len(test_history[test_id]) == 0:
            return np.zeros(len(self.feature_names))
        
        history = test_history[test_id]
        recent_history = history[-window_size:]
        
        outcomes = [h['outcome'] for h in recent_history]
        exec_times = [h['execution_time'] for h in recent_history]
        
        recent_pass_rate = sum(1 - o for o in outcomes) / len(outcomes) if outcomes else 0.5
        
        avg_exec_time = np.mean(exec_times) if exec_times else 1.0
        
        failure_streak = 0
        for outcome in reversed(outcomes):
            if outcome == 1:
                failure_streak += 1
            else:
                break
        
        coverage_overlap = commit_data.get('coverage_overlap', 0.0)
        
        time_since_last_fail = 0
        for i, outcome in enumerate(reversed(outcomes)):
            if outcome == 1:
                time_since_last_fail = i
                break
        else:
            time_since_last_fail = len(outcomes)
        
        exec_time_variance = np.var(exec_times) if len(exec_times) > 1 else 0.0
        
        files_modified = commit_data.get('files_modified', 0)
        lines_added = commit_data.get('lines_added', 0)
        lines_deleted = commit_data.get('lines_deleted', 0)
        code_churn = lines_added + lines_deleted
        
        token_count = commit_data.get('token_count', lines_added * 8 + lines_deleted * 4)
        method_count_delta = commit_data.get('method_count_delta', files_modified // 3)
        syntactic_complexity = commit_data.get('syntactic_complexity', 
                                               code_churn / max(files_modified, 1))
        
        features = np.array([
            files_modified,
            lines_added,
            lines_deleted,
            code_churn,
            recent_pass_rate,
            avg_exec_time,
            failure_streak,
            coverage_overlap,
            time_since_last_fail,
            exec_time_variance,
            token_count,
            method_count_delta,
            syntactic_complexity
        ])
        
        return features
    
    def extract_features_batch(self, test_ids, test_history, commit_data, window_size=10):
        features = np.array([
            self.extract_test_features(tid, test_history, commit_data, window_size)
            for tid in test_ids
        ])
        return features
    
    def get_feature_names(self):
        return self.feature_names.copy()


class DataSimulator:
    def __init__(self, n_tests=100, n_commits=50, failure_rate=0.15, seed=42):
        self.n_tests = n_tests
        self.n_commits = n_commits
        self.failure_rate = failure_rate
        self.seed = seed
        
    def generate_test_suite(self):
        rng = np.random.RandomState(self.seed)
        tests = []
        for i in range(self.n_tests):
            test = {
                'test_id': f'test_{i:04d}',
                'name': f'Test_{chr(65 + i % 26)}_{i // 26}',
                'module': f'module_{i % 10}',
                'execution_time': rng.lognormal(1.5, 0.8),
                'base_failure_prob': rng.beta(2, 10) * self.failure_rate / 0.15,
                'flakiness': rng.beta(1, 20)
            }
            tests.append(test)
        return pd.DataFrame(tests)
    
    def generate_commits(self):
        rng = np.random.RandomState(self.seed + 1)
        commits = []
        for i in range(self.n_commits):
            files_modified = rng.poisson(5) + 1
            lines_added = rng.poisson(50)
            lines_deleted = rng.poisson(30)
            code_churn = lines_added + lines_deleted
            commit = {
                'commit_id': f'commit_{i:04d}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i*6),
                'files_modified': files_modified,
                'lines_added': lines_added,
                'lines_deleted': lines_deleted,
                'token_count': lines_added * 8 + lines_deleted * 4,
                'method_count_delta': files_modified // 3,
                'syntactic_complexity': code_churn / max(files_modified, 1),
                'author': f'dev_{i % 5}',
                'message': f'Feature update {i}'
            }
            commits.append(commit)
        return pd.DataFrame(commits)
    
    def simulate_test_execution(self, test_suite, commits):
        rng = np.random.RandomState(self.seed + 2)
        test_history = {test_id: [] for test_id in test_suite['test_id']}
        execution_log = []
        
        for _, commit in commits.iterrows():
            for _, test in test_suite.iterrows():
                base_prob = test['base_failure_prob']
                
                commit_impact = (commit['files_modified'] / 20.0) * 0.3
                
                failure_prob = min(base_prob + commit_impact, 0.9)
                
                if rng.random() < test['flakiness']:
                    outcome = rng.choice([0, 1])
                else:
                    outcome = 1 if rng.random() < failure_prob else 0
                
                exec_time = test['execution_time'] * (1 + rng.normal(0, 0.1))
                
                record = {
                    'outcome': outcome,
                    'execution_time': max(exec_time, 0.1),
                    'commit_id': commit['commit_id']
                }
                
                test_history[test['test_id']].append(record)
                
                execution_log.append({
                    'commit_id': commit['commit_id'],
                    'test_id': test['test_id'],
                    'outcome': outcome,
                    'execution_time': record['execution_time'],
                    'timestamp': commit['timestamp']
                })
        
        return test_history, pd.DataFrame(execution_log)
