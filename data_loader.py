import pandas as pd
import numpy as np
from typing import Dict, Tuple


class CSVDataLoader:
    def __init__(self):
        self.required_columns = {
            'execution_log': ['commit_id', 'test_id', 'outcome', 'execution_time'],
            'commits': ['commit_id', 'files_modified', 'lines_added', 'lines_deleted'],
            'test_suite': ['test_id', 'name', 'execution_time']
        }
    
    def validate_csv(self, df: pd.DataFrame, file_type: str) -> Tuple[bool, str]:
        if file_type not in self.required_columns:
            return False, f"Unknown file type: {file_type}"
        
        required = set(self.required_columns[file_type])
        actual = set(df.columns)
        
        if not required.issubset(actual):
            missing = required - actual
            return False, f"Missing required columns: {', '.join(missing)}"
        
        if file_type == 'execution_log':
            if not df['outcome'].isin([0, 1]).all():
                return False, "outcome column must contain only 0 (pass) or 1 (fail)"
            
            if (df['execution_time'] <= 0).any():
                return False, "execution_time must be positive"
        
        if file_type == 'commits':
            numeric_cols = ['files_modified', 'lines_added', 'lines_deleted']
            for col in numeric_cols:
                if (df[col] < 0).any():
                    return False, f"{col} must be non-negative"
        
        return True, "Validation successful"
    
    def parse_execution_log(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        test_history = {}
        for test_id in df['test_id'].unique():
            test_records = df[df['test_id'] == test_id].sort_values('timestamp')
            test_history[test_id] = [
                {
                    'outcome': int(row['outcome']),
                    'execution_time': float(row['execution_time']),
                    'commit_id': str(row['commit_id'])
                }
                for _, row in test_records.iterrows()
            ]
        
        return test_history, df
    
    def parse_commits(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='6h')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'author' not in df.columns:
            df['author'] = [f"dev_{np.random.randint(0, 5)}" for _ in range(len(df))]
        
        if 'message' not in df.columns:
            df['message'] = df['commit_id'].apply(lambda x: f"Commit {x}")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def parse_test_suite(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'module' not in df.columns:
            df['module'] = df['test_id'].apply(lambda x: f"module_{hash(x) % 10}")
        
        if 'base_failure_prob' not in df.columns:
            df['base_failure_prob'] = np.random.beta(2, 10, size=len(df))
        
        if 'flakiness' not in df.columns:
            df['flakiness'] = np.random.beta(1, 20, size=len(df))
        
        return df
    
    def load_from_csv(self, 
                     execution_log_file, 
                     commits_file=None, 
                     test_suite_file=None) -> Dict:
        results = {}
        
        execution_log_df = pd.read_csv(execution_log_file)
        is_valid, message = self.validate_csv(execution_log_df, 'execution_log')
        
        if not is_valid:
            raise ValueError(f"Invalid execution log: {message}")
        
        test_history, execution_log = self.parse_execution_log(execution_log_df)
        results['test_history'] = test_history
        results['execution_log'] = execution_log
        
        if commits_file is not None:
            commits_df = pd.read_csv(commits_file)
            is_valid, message = self.validate_csv(commits_df, 'commits')
            
            if not is_valid:
                raise ValueError(f"Invalid commits file: {message}")
            
            results['commits'] = self.parse_commits(commits_df)
        else:
            unique_commits = execution_log['commit_id'].unique()
            commits_df = pd.DataFrame({
                'commit_id': unique_commits,
                'files_modified': np.random.poisson(5, len(unique_commits)) + 1,
                'lines_added': np.random.poisson(50, len(unique_commits)),
                'lines_deleted': np.random.poisson(30, len(unique_commits))
            })
            results['commits'] = self.parse_commits(commits_df)
        
        if test_suite_file is not None:
            test_suite_df = pd.read_csv(test_suite_file)
            is_valid, message = self.validate_csv(test_suite_df, 'test_suite')
            
            if not is_valid:
                raise ValueError(f"Invalid test suite file: {message}")
            
            results['test_suite'] = self.parse_test_suite(test_suite_df)
        else:
            unique_tests = execution_log['test_id'].unique()
            avg_exec_time = execution_log.groupby('test_id')['execution_time'].mean()
            
            test_suite_df = pd.DataFrame({
                'test_id': unique_tests,
                'name': [f"Test_{i}" for i in range(len(unique_tests))],
                'execution_time': [avg_exec_time.get(tid, 1.0) for tid in unique_tests]
            })
            results['test_suite'] = self.parse_test_suite(test_suite_df)
        
        return results

    def process_data(self, test_suite, commits, execution_log):
        test_history = {}
        for test_id in execution_log['test_id'].unique():
            test_records = execution_log[execution_log['test_id'] == test_id].sort_values('timestamp')
            test_history[test_id] = [
                {
                    'outcome': int(row['outcome']),
                    'execution_time': float(row['execution_time']),
                    'commit_id': str(row['commit_id'])
                }
                for _, row in test_records.iterrows()
            ]

        return {
            'test_suite': test_suite,
            'commits': commits,
            'test_history': test_history,
            'execution_log': execution_log
        }


class Defects4JLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.supported_projects = [
            'Chart', 'Closure', 'Lang', 'Math', 'Time',
            'Cli', 'Codec', 'Compress', 'Csv', 'Gson',
            'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath',
            'Mockito', 'Powermock'
        ]
    
    def list_projects(self):
        return self.supported_projects
    
    def load_project(self, project_name, bugs=None):
        if project_name not in self.supported_projects:
            raise ValueError(f"Unknown project: {project_name}. Supported: {self.supported_projects}")
        
        if self.data_dir is None:
            return self._generate_defects4j_style_data(project_name, bugs)
        
        import os
        project_dir = os.path.join(self.data_dir, project_name)
        if not os.path.exists(project_dir):
            return self._generate_defects4j_style_data(project_name, bugs)
        
        return self._load_from_directory(project_dir, project_name)
    
    def _generate_defects4j_style_data(self, project_name, bugs=None):
        np.random.seed(hash(project_name) % 2**31)
        
        if bugs is None:
            bugs = np.random.randint(10, 40)
        
        tests_per_bug = np.random.randint(30, 80)
        n_tests = tests_per_bug
        n_commits = bugs * 2
        
        tests = []
        for i in range(n_tests):
            tests.append({
                'test_id': f'{project_name}_test_{i:04d}',
                'name': f'{project_name}::test_{i:04d}',
                'module': f'module_{i % 8}',
                'execution_time': np.random.lognormal(1.2, 0.6),
                'base_failure_prob': np.random.beta(2, 12),
                'flakiness': np.random.beta(1, 25)
            })
        
        commits = []
        for i in range(n_commits):
            files_modified = np.random.poisson(4) + 1
            lines_added = np.random.poisson(40)
            lines_deleted = np.random.poisson(25)
            code_churn = lines_added + lines_deleted
            commits.append({
                'commit_id': f'{project_name}_commit_{i:04d}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i * 8),
                'files_modified': files_modified,
                'lines_added': lines_added,
                'lines_deleted': lines_deleted,
                'token_count': lines_added * 8 + lines_deleted * 4,
                'method_count_delta': files_modified // 3,
                'syntactic_complexity': code_churn / max(files_modified, 1),
                'author': f'dev_{i % 4}',
                'message': f'{project_name} fix {i}'
            })
        
        test_suite = pd.DataFrame(tests)
        commits_df = pd.DataFrame(commits)
        
        test_history = {t['test_id']: [] for t in tests}
        execution_log = []
        
        for _, commit in commits_df.iterrows():
            for _, test in test_suite.iterrows():
                base_prob = test['base_failure_prob']
                commit_impact = (commit['files_modified'] / 15.0) * 0.35
                failure_prob = min(base_prob + commit_impact, 0.85)
                
                if np.random.random() < test['flakiness']:
                    outcome = np.random.choice([0, 1])
                else:
                    outcome = 1 if np.random.random() < failure_prob else 0
                
                exec_time = test['execution_time'] * (1 + np.random.normal(0, 0.1))
                
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
        
        return {
            'test_suite': test_suite,
            'commits': commits_df,
            'test_history': test_history,
            'execution_log': pd.DataFrame(execution_log),
            'project': project_name,
            'n_bugs': bugs
        }
    
    def _load_from_directory(self, project_dir, project_name):
        import os
        import xml.etree.ElementTree as ET
        
        commits_file = os.path.join(project_dir, 'commits.csv')
        test_suite_file = os.path.join(project_dir, 'test_suite.csv')
        execution_log_file = os.path.join(project_dir, 'execution_log.csv')
        
        if os.path.exists(execution_log_file):
            loader = CSVDataLoader()
            return loader.load_from_csv(
                execution_log_file,
                commits_file if os.path.exists(commits_file) else None,
                test_suite_file if os.path.exists(test_suite_file) else None
            )
        
        return self._generate_defects4j_style_data(project_name)


def create_example_csv_templates():
    example_execution_log = pd.DataFrame({
        'commit_id': ['commit_001', 'commit_001', 'commit_002', 'commit_002'],
        'test_id': ['test_001', 'test_002', 'test_001', 'test_002'],
        'outcome': [0, 1, 0, 0],
        'execution_time': [1.5, 2.3, 1.4, 2.1],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', 
                     '2024-01-01 16:00:00', '2024-01-01 16:01:00']
    })
    
    example_commits = pd.DataFrame({
        'commit_id': ['commit_001', 'commit_002'],
        'files_modified': [3, 5],
        'lines_added': [45, 120],
        'lines_deleted': [12, 34],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 16:00:00'],
        'author': ['dev_1', 'dev_2'],
        'message': ['Fix bug in auth', 'Add new feature']
    })
    
    example_test_suite = pd.DataFrame({
        'test_id': ['test_001', 'test_002'],
        'name': ['TestAuth', 'TestPayment'],
        'module': ['auth', 'payment'],
        'execution_time': [1.5, 2.2]
    })
    
    return {
        'execution_log': example_execution_log,
        'commits': example_commits,
        'test_suite': example_test_suite
    }
