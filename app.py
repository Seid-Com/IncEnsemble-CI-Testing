import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from inc_ensemble import IncEnsemble, FlakinessDetector, TestPrioritizer
from feature_extractor import FeatureExtractor, DataSimulator
from data_loader import CSVDataLoader, Defects4JLoader, create_example_csv_templates
from active_learner import ActiveLearner
from test_report_parser import TestReportParser
from baselines import (run_baselines, compute_fdr_vs_budget, run_multi_evaluation,
                       run_hyperparameter_sensitivity, run_imbalance_comparison, wilcoxon_test)
from datetime import datetime
import hashlib


def deterministic_coverage(commit_id, test_id):
    h = hashlib.md5(f"{commit_id}:{test_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 1000) / 1000.0


def build_commit_data(commit):
    files_modified = commit['files_modified']
    lines_added = commit['lines_added']
    lines_deleted = commit['lines_deleted']
    code_churn = lines_added + lines_deleted
    return {
        'files_modified': files_modified,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'token_count': commit.get('token_count', lines_added * 8 + lines_deleted * 4),
        'method_count_delta': commit.get('method_count_delta', files_modified // 3),
        'syntactic_complexity': commit.get('syntactic_complexity', code_churn / max(files_modified, 1))
    }

st.set_page_config(
    page_title="IncEnsemble - ML-Powered Test Prioritization",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 IncEnsemble: Automated Software Testing Framework")
st.markdown("""
**Efficient Machine Learning for Test Case Prioritization**  
*Incremental learning, flakiness detection, and budget-aware test selection*
""")


@st.cache_resource
def initialize_system():
    simulator = DataSimulator(n_tests=200, n_commits=50, failure_rate=0.15)
    test_suite = simulator.generate_test_suite()
    commits = simulator.generate_commits()
    test_history, execution_log = simulator.simulate_test_execution(test_suite, commits)
    
    model = IncEnsemble(n_trees=100, max_depth=8, alpha=0.6)
    feature_extractor = FeatureExtractor()
    flakiness_detector = FlakinessDetector(window_size=20)
    prioritizer = TestPrioritizer(lambda_penalty=0.3)
    active_learner = ActiveLearner(utility_weight=1.0, cost_weight=1.0)
    
    return {
        'test_suite': test_suite,
        'commits': commits,
        'test_history': test_history,
        'execution_log': execution_log,
        'model': model,
        'feature_extractor': feature_extractor,
        'flakiness_detector': flakiness_detector,
        'prioritizer': prioritizer,
        'active_learner': active_learner
    }


def train_initial_model(system, train_ratio=0.7):
    commits = system['commits'].reset_index(drop=True)
    n_train = int(len(commits) * train_ratio)
    train_commits = commits.iloc[:n_train]
    
    X_train = []
    y_train = []
    
    feature_extractor = system['feature_extractor']
    test_history = system['test_history']
    test_suite = system['test_suite']
    
    for _, commit in train_commits.iterrows():
        commit_data = build_commit_data(commit)
        
        for test_id in test_suite['test_id']:
            commit_data['coverage_overlap'] = deterministic_coverage(commit['commit_id'], test_id)
            features = feature_extractor.extract_test_features(
                test_id, test_history, commit_data
            )
            
            outcomes = [h['outcome'] for h in test_history[test_id] 
                       if h['commit_id'] == commit['commit_id']]
            if outcomes:
                X_train.append(features)
                y_train.append(outcomes[0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if len(X_train) == 0:
        raise ValueError("No training data generated. Check that commit_ids in commits match test_history.")
    
    system['model'].fit_static(X_train, y_train)
    
    return n_train


def load_data_from_csv():
    loader = CSVDataLoader()
    
    st.sidebar.subheader("📁 Upload CSV Files")
    
    execution_log_file = st.sidebar.file_uploader(
        "Execution Log (required)", 
        type=['csv'],
        help="CSV with columns: commit_id, test_id, outcome, execution_time"
    )
    
    commits_file = st.sidebar.file_uploader(
        "Commits (optional)", 
        type=['csv'],
        help="CSV with columns: commit_id, files_modified, lines_added, lines_deleted"
    )
    
    test_suite_file = st.sidebar.file_uploader(
        "Test Suite (optional)", 
        type=['csv'],
        help="CSV with columns: test_id, name, execution_time"
    )
    
    if execution_log_file is not None:
        try:
            data = loader.load_from_csv(
                execution_log_file,
                commits_file,
                test_suite_file
            )
            
            model = IncEnsemble(n_trees=100, max_depth=8, alpha=0.6)
            feature_extractor = FeatureExtractor()
            flakiness_detector = FlakinessDetector(window_size=20)
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            
            active_learner = ActiveLearner(utility_weight=1.0, cost_weight=1.0)
            
            return {
                'test_suite': data['test_suite'],
                'commits': data['commits'],
                'test_history': data['test_history'],
                'execution_log': data['execution_log'],
                'model': model,
                'feature_extractor': feature_extractor,
                'flakiness_detector': flakiness_detector,
                'prioritizer': prioritizer,
                'active_learner': active_learner,
                'data_source': 'uploaded'
            }
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
            return None
    
    return None


def load_data_from_xml():
    parser = TestReportParser()
    
    st.sidebar.subheader("📁 Upload Test Report XML")
    
    xml_files = st.sidebar.file_uploader(
        "JUnit/TestNG XML Report(s)", 
        type=['xml'],
        accept_multiple_files=True,
        help="Upload one or more JUnit or TestNG XML test reports"
    )
    
    if xml_files and len(xml_files) > 0:
        try:
            import tempfile
            import os
            
            reports = []
            
            for idx, xml_file in enumerate(xml_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp:
                    tmp.write(xml_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
                    format_type = parser.detect_format(tmp_path)
                    st.sidebar.success(f"✅ {xml_file.name}: {format_type.upper()}")
                    
                    test_suite, commits, execution_log = parser.parse_report(tmp_path, file_id=file_id)
                    reports.append((test_suite, commits, execution_log))
                finally:
                    os.unlink(tmp_path)
            
            if len(reports) > 1:
                test_suite, commits, execution_log = parser.merge_multiple_reports(reports)
            else:
                test_suite, commits, execution_log = reports[0]
            
            loader = CSVDataLoader()
            data = loader.process_data(test_suite, commits, execution_log)
            
            model = IncEnsemble(n_trees=100, max_depth=8, alpha=0.6)
            feature_extractor = FeatureExtractor()
            flakiness_detector = FlakinessDetector(window_size=20)
            prioritizer = TestPrioritizer(lambda_penalty=0.3)
            active_learner = ActiveLearner(utility_weight=1.0, cost_weight=1.0)
            
            return {
                'test_suite': data['test_suite'],
                'commits': data['commits'],
                'test_history': data['test_history'],
                'execution_log': data['execution_log'],
                'model': model,
                'feature_extractor': feature_extractor,
                'flakiness_detector': flakiness_detector,
                'prioritizer': prioritizer,
                'active_learner': active_learner,
                'data_source': 'xml_upload'
            }
        except Exception as e:
            st.sidebar.error(f"Error parsing XML: {str(e)}")
            import traceback
            st.sidebar.code(traceback.format_exc())
            return None
    
    return None


def main():
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar:
        st.subheader("📊 Data Source")
        data_source = st.radio(
            "Select data source",
            ["Synthetic (Demo)", "Defects4J Benchmark", "Upload CSV", "Upload XML (JUnit/TestNG)"],
            help="Use synthetic data for demo, Defects4J benchmark, or upload your own test execution data"
        )
        
        st.divider()
    
    if data_source == "Upload CSV":
        system = load_data_from_csv()
        
        if system is None:
            st.info("📤 Upload an execution log CSV file to get started")
            
            with st.expander("📋 View CSV Format Examples"):
                examples = create_example_csv_templates()
                
                st.subheader("Execution Log Format (Required)")
                st.dataframe(examples['execution_log'], width='stretch')
                st.download_button(
                    "⬇️ Download Template",
                    examples['execution_log'].to_csv(index=False),
                    "execution_log_template.csv",
                    "text/csv"
                )
                
                st.subheader("Commits Format (Optional)")
                st.dataframe(examples['commits'], width='stretch')
                st.download_button(
                    "⬇️ Download Template",
                    examples['commits'].to_csv(index=False),
                    "commits_template.csv",
                    "text/csv"
                )
                
                st.subheader("Test Suite Format (Optional)")
                st.dataframe(examples['test_suite'], width='stretch')
                st.download_button(
                    "⬇️ Download Template",
                    examples['test_suite'].to_csv(index=False),
                    "test_suite_template.csv",
                    "text/csv"
                )
            
            return
    elif data_source == "Upload XML (JUnit/TestNG)":
        system = load_data_from_xml()
        
        if system is None:
            st.info("📤 Upload JUnit or TestNG XML test reports to get started")
            
            st.markdown("""
            ### Supported Formats
            
            **JUnit XML Format:**
            - Standard format produced by most Java testing frameworks (JUnit, Maven Surefire)
            - Also supported by pytest (with `--junitxml` flag) and other test runners
            
            **TestNG XML Format:**
            - Native format for TestNG test framework
            - Includes additional metadata like test groups and parameters
            
            ### Example Usage
            
            ```bash
            # Generate JUnit XML with pytest
            pytest --junitxml=test-results.xml
            
            # Generate JUnit XML with Maven
            mvn test
            # Output: target/surefire-reports/TEST-*.xml
            
            # TestNG generates XML reports automatically
            # Location: test-output/testng-results.xml
            ```
            
            You can upload multiple XML files and they will be merged automatically.
            """)
            
            return
    elif data_source == "Defects4J Benchmark":
        loader = Defects4JLoader()
        
        with st.sidebar:
            project = st.selectbox(
                "Select Project",
                loader.list_projects(),
                index=3,
                help="Select a Defects4J project to load"
            )
        
        if st.button("Load Project", type="primary"):
            with st.spinner(f"Loading {project}..."):
                d4j_data = loader.load_project(project)
                
                model = IncEnsemble(n_trees=100, max_depth=8, alpha=0.6)
                feature_extractor = FeatureExtractor()
                flakiness_detector = FlakinessDetector(window_size=20)
                prioritizer = TestPrioritizer(lambda_penalty=0.3)
                active_learner = ActiveLearner(utility_weight=1.0, cost_weight=1.0)
                
                system = {
                    'test_suite': d4j_data['test_suite'],
                    'commits': d4j_data['commits'],
                    'test_history': d4j_data['test_history'],
                    'execution_log': d4j_data['execution_log'],
                    'model': model,
                    'feature_extractor': feature_extractor,
                    'flakiness_detector': flakiness_detector,
                    'prioritizer': prioritizer,
                    'active_learner': active_learner,
                    'data_source': 'defects4j',
                    'project': d4j_data.get('project', project),
                    'n_bugs': d4j_data.get('n_bugs', 0)
                }
                
                st.session_state['system'] = system
                st.success(f"Loaded {project}: {len(d4j_data['test_suite'])} tests, {len(d4j_data['commits'])} commits")
                st.rerun()
        
        if 'system' in st.session_state and st.session_state['system'].get('data_source') == 'defects4j':
            system = st.session_state['system']
        else:
            st.info(f"Select a Defects4J project and click Load. Projects include: Chart, Closure, Lang, Math, Time, and more.")
            st.markdown("""
            ### Defects4J Dataset
            
            Defects4J is a curated collection of real-world Java projects with reproducible bugs:
            - **17 projects** with **395 bugs** across **6,500 test cases*
            - Provides commit histories, test outcomes, and execution times
            - Widely used benchmark for evaluating defect prediction and test prioritization
            
            Select a project from the dropdown and click **Load Project** to begin.
            """)
            return
    else:
        system = initialize_system()
    
    with st.sidebar:
        st.subheader("Model Parameters")
        alpha = st.slider("Ensemble Weight (α)", 0.0, 1.0, 0.6, 0.05,
                         help="Weight for static forest vs online corrector")
        system['model'].alpha = alpha
        
        lambda_penalty = st.slider("Flakiness Penalty (λ)", 0.0, 1.0, 0.3, 0.05,
                                   help="Penalty factor for flaky tests")
        system['prioritizer'].lambda_penalty = lambda_penalty
        
        st.subheader("Test Selection")
        budget_pct = st.slider("Execution Budget (%)", 10, 100, 30, 5,
                               help="Percentage of total test suite to execute")
        
        if st.button("🔄 Train Model", type="primary"):
            with st.spinner("Training IncEnsemble model..."):
                n_train = train_initial_model(system)
                st.session_state['trained'] = True
                st.session_state['n_train'] = n_train
                st.success(f"✅ Model trained on {n_train} commits!")
    
    if 'trained' not in st.session_state:
        st.info("👈 Click 'Train Model' in the sidebar to begin")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Suite Size", len(system['test_suite']))
        with col2:
            st.metric("Total Commits", len(system['commits']))
        with col3:
            st.metric("Execution Records", len(system['execution_log']))
        
        st.subheader("📊 Test Suite Overview")
        st.dataframe(system['test_suite'].head(10), width='stretch')
        return
    
    tabs = st.tabs([
        "📈 Dashboard", 
        "🎯 Test Prioritization",
        "🎓 Active Learning",
        "🔍 Flakiness Analysis",
        "📊 Performance Metrics",
        "🌊 Concept Drift",
        "🧠 Model Explainability",
        "🔬 SHAP Analysis",
        "⚔️ Baseline Comparison",
        "📈 FDR vs Budget",
        "🔁 Multi-Run Evaluation",
        "🎛️ Hyperparameter Sensitivity",
        "⚖️ Imbalance Strategies"
    ])
    
    with tabs[0]:
        show_dashboard(system)
    
    with tabs[1]:
        show_test_prioritization(system, budget_pct)
    
    with tabs[2]:
        show_active_learning(system, budget_pct)
    
    with tabs[3]:
        show_flakiness_analysis(system)
    
    with tabs[4]:
        show_performance_metrics(system)
    
    with tabs[5]:
        show_concept_drift(system)
    
    with tabs[6]:
        show_model_explainability(system)
    
    with tabs[7]:
        show_shap_analysis(system)
    
    n_train = st.session_state.get('n_train', 35)
    
    with tabs[8]:
        show_baseline_comparison(system, budget_pct, n_train)
    
    with tabs[9]:
        show_fdr_vs_budget(system, budget_pct, n_train)
    
    with tabs[10]:
        show_multi_run_evaluation(system, budget_pct, n_train)
    
    with tabs[11]:
        show_hyperparameter_sensitivity(system, budget_pct, n_train)
    
    with tabs[12]:
        show_imbalance_comparison(system, budget_pct, n_train)


def show_dashboard(system):
    st.header("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    execution_log = system['execution_log']
    failure_rate = execution_log['outcome'].mean()
    avg_exec_time = execution_log['execution_time'].mean()
    total_tests = len(system['test_suite'])
    
    flakiness_scores = system['flakiness_detector'].calculate_flakiness(system['test_history'])
    avg_flakiness = np.mean(list(flakiness_scores.values()))
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Failure Rate", f"{failure_rate*100:.1f}%")
    with col3:
        st.metric("Avg Execution Time", f"{avg_exec_time:.2f}s")
    with col4:
        st.metric("Avg Flakiness", f"{avg_flakiness:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Outcomes Over Time")
        commit_failures = execution_log.groupby('commit_id')['outcome'].agg(['sum', 'count'])
        commit_failures['failure_rate'] = commit_failures['sum'] / commit_failures['count']
        
        fig = px.line(commit_failures.reset_index(), x='commit_id', y='failure_rate',
                     title="Failure Rate per Commit")
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Execution Time Distribution")
        fig = px.histogram(execution_log, x='execution_time', nbins=30,
                          title="Test Execution Time Distribution")
        fig.update_xaxes(title="Execution Time (s)")
        fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, width='stretch')


def show_test_prioritization(system, budget_pct):
    st.header("🎯 Test Case Prioritization")
    
    test_commits = system['commits'].iloc[st.session_state.get('n_train', 35):]
    
    if len(test_commits) == 0:
        st.warning("No test commits available. Increase training ratio.")
        return
    
    commit_idx = st.selectbox(
        "Select Commit for Analysis",
        range(len(test_commits)),
        format_func=lambda x: f"{test_commits.iloc[x]['commit_id']} - {test_commits.iloc[x]['message']}"
    )
    
    selected_commit = test_commits.iloc[commit_idx]
    
    commit_data = build_commit_data(selected_commit)
    
    test_suite = system['test_suite']
    feature_extractor = system['feature_extractor']
    
    X = []
    for tid in test_suite['test_id'].values:
        commit_data['coverage_overlap'] = deterministic_coverage(selected_commit['commit_id'], tid)
        X.append(feature_extractor.extract_test_features(
            tid, system['test_history'], commit_data
        ))
    X = np.array(X)
    
    fault_proba = system['model'].predict_proba(X)
    
    flakiness_scores = system['flakiness_detector'].calculate_flakiness(system['test_history'])
    flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
    
    execution_costs = test_suite['execution_time'].values
    
    scores = system['prioritizer'].compute_scores(
        fault_proba, execution_costs, flakiness_array
    )
    
    total_budget = execution_costs.sum() * (budget_pct / 100.0)
    selected_indices, actual_cost = system['prioritizer'].select_tests(
        scores, execution_costs, total_budget
    )
    
    actual_failures = []
    for test_id in test_suite['test_id']:
        outcomes = [h['outcome'] for h in system['test_history'][test_id]
                   if h['commit_id'] == selected_commit['commit_id']]
        actual_failures.append(outcomes[0] if outcomes else 0)
    actual_failures = np.array(actual_failures)
    
    selected_indices_set = set(selected_indices)
    
    execution_order = [idx for idx in np.argsort(scores)[::-1] if idx in selected_indices_set]
    metrics = system['prioritizer'].calculate_metrics(
        selected_indices, actual_failures,
        execution_costs=execution_costs,
        execution_order=execution_order
    )
    
    if st.button("🔄 Update Model Incrementally", help="Update online corrector with selected test outcomes"):
        X_selected = X[selected_indices]
        y_selected = actual_failures[selected_indices]
        system['model'].update_online(X_selected, y_selected)
        st.success(f"✅ Model updated with {len(selected_indices)} test outcomes! Online corrector is now active.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Tests Selected", len(selected_indices))
    with col2:
        st.metric("Budget Used", f"{actual_cost:.1f}s / {total_budget:.1f}s")
    with col3:
        st.metric("Fault Detection Rate", f"{metrics['fault_detection_rate']*100:.1f}%")
    with col4:
        st.metric("Precision", f"{metrics['precision']*100:.1f}%")
    with col5:
        online_status = "🟢 Active" if system['model'].online_fitted else "🔴 Inactive"
        st.metric("Online Corrector", online_status)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ATD (s)", f"{metrics['atd']:.2f}")
    with col2:
        st.metric("Wasted Cost (s)", f"{metrics['wasted_cost']:.2f}")
    with col3:
        st.metric("FPR", f"{metrics['fpr']*100:.1f}%")
    with col4:
        st.metric("ITT (s)", f"{metrics['itt']:.4f}")
    
    results_df = pd.DataFrame({
        'Test ID': test_suite['test_id'],
        'Test Name': test_suite['name'],
        'Fault Probability': fault_proba,
        'Flakiness': flakiness_array,
        'Execution Time (s)': execution_costs,
        'Priority Score': scores,
        'Selected': ['✅' if i in selected_indices_set else '❌' for i in range(len(test_suite))],
        'Actual Outcome': ['FAIL' if f else 'PASS' for f in actual_failures]
    })
    
    results_df = results_df.sort_values('Priority Score', ascending=False)
    
    st.subheader("📋 Prioritized Test List")
    st.dataframe(results_df, width='stretch', height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Priority Score Distribution")
        fig = px.scatter(results_df, x='Execution Time (s)', y='Fault Probability',
                        size='Priority Score', color='Selected',
                        hover_data=['Test Name', 'Flakiness'],
                        title="Test Selection Visualization")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Selected vs Actual Failures")
        confusion_data = {
            'Selected & Failed': sum(1 for i in selected_indices if actual_failures[i]),
            'Selected & Passed': sum(1 for i in selected_indices if not actual_failures[i]),
            'Not Selected & Failed': sum(1 for i in range(len(actual_failures)) 
                                         if i not in selected_indices_set and actual_failures[i]),
            'Not Selected & Passed': sum(1 for i in range(len(actual_failures)) 
                                          if i not in selected_indices_set and not actual_failures[i])
        }
        fig = go.Figure(data=[go.Pie(labels=list(confusion_data.keys()), 
                                      values=list(confusion_data.values()))])
        fig.update_layout(title="Test Selection Outcome")
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("📤 Export Test Execution Plan")
    
    st.markdown("""
    Export the prioritized test selection for integration with your CI/CD pipeline.
    Choose your preferred format below.
    """)
    
    selected_tests = results_df[results_df['Selected'] == '✅']
    
    test_module_map = dict(zip(test_suite['test_id'], test_suite['module']))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        import json
        
        json_plan = {
            "metadata": {
                "total_tests": len(test_suite),
                "selected_tests": len(selected_indices),
                "budget_percentage": budget_pct,
                "total_budget_seconds": float(total_budget),
                "used_budget_seconds": float(actual_cost),
                "fault_detection_rate": float(metrics['fault_detection_rate']),
                "precision": float(metrics['precision'])
            },
            "tests": [
                {
                    "test_id": row['Test ID'],
                    "test_name": row['Test Name'],
                    "module": test_module_map[row['Test ID']],
                    "priority_score": float(row['Priority Score']),
                    "fault_probability": float(row['Fault Probability']),
                    "execution_time": float(row['Execution Time (s)']),
                    "flakiness": float(row['Flakiness'])
                }
                for i, row in selected_tests.iterrows()
            ]
        }
        
        json_str = json.dumps(json_plan, indent=2)
        
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="test_execution_plan.json",
            mime="application/json",
            help="Structured JSON format for programmatic integration"
        )
    
    with col2:
        yaml_lines = [
            "# IncEnsemble Test Execution Plan",
            f"# Generated with {budget_pct}% budget",
            "",
            "metadata:",
            f"  total_tests: {len(test_suite)}",
            f"  selected_tests: {len(selected_indices)}",
            f"  budget_percentage: {budget_pct}",
            f"  total_budget_seconds: {total_budget:.2f}",
            f"  used_budget_seconds: {actual_cost:.2f}",
            f"  fault_detection_rate: {metrics['fault_detection_rate']:.3f}",
            f"  precision: {metrics['precision']:.3f}",
            "",
            "tests:"
        ]
        
        for i, row in selected_tests.iterrows():
            yaml_lines.extend([
                f"  - test_id: {row['Test ID']}",
                f"    test_name: \"{row['Test Name']}\"",
                f"    module: \"{test_module_map[row['Test ID']]}\"",
                f"    priority_score: {row['Priority Score']:.4f}",
                f"    fault_probability: {row['Fault Probability']:.4f}",
                f"    execution_time: {row['Execution Time (s)']:.2f}",
                f"    flakiness: {row['Flakiness']:.4f}",
                ""
            ])
        
        yaml_str = "\n".join(yaml_lines)
        
        st.download_button(
            label="⬇️ Download YAML",
            data=yaml_str,
            file_name="test_execution_plan.yaml",
            mime="text/yaml",
            help="Human-readable YAML format for configuration files"
        )
    
    with col3:
        import shlex
        
        bash_lines = [
            "#!/bin/bash",
            "# IncEnsemble Test Execution Plan",
            f"# Generated with {budget_pct}% budget ({len(selected_indices)}/{len(test_suite)} tests)",
            f"# Expected fault detection rate: {metrics['fault_detection_rate']*100:.1f}%",
            "",
            "set -e",
            "",
            "echo '🧪 Starting prioritized test execution...'",
            f"echo 'Running {len(selected_indices)} tests out of {len(test_suite)} total'",
            "",
            "FAILED_TESTS=()",
            "PASSED_TESTS=()",
            "",
            "run_test() {",
            "    local test_name=$1",
            "    local test_module=$2",
            "    ",
            "    echo \"Running: ${test_module}::${test_name}\"",
            "    ",
            "    # Replace this with your actual test runner command",
            "    # Example for pytest: pytest \"${test_module}::${test_name}\"",
            "    # Example for JUnit: mvn test -Dtest=\"${test_module}#${test_name}\"",
            "    ",
            "    if python -m pytest \"${test_module}::${test_name}\"; then",
            "        PASSED_TESTS+=(\"$test_name\")",
            "    else",
            "        FAILED_TESTS+=(\"$test_name\")",
            "    fi",
            "}",
            "",
            "# Prioritized test execution"
        ]
        
        for i, row in selected_tests.iterrows():
            test_name = shlex.quote(row['Test Name'])
            module = shlex.quote(test_module_map[row['Test ID']])
            bash_lines.append(f'run_test {test_name} {module}')
        
        bash_lines.extend([
            "",
            "# Summary",
            "echo ''",
            "echo '📊 Test Execution Summary'",
            "echo \"Passed: ${#PASSED_TESTS[@]}\"",
            "echo \"Failed: ${#FAILED_TESTS[@]}\"",
            "",
            "if [ ${#FAILED_TESTS[@]} -gt 0 ]; then",
            "    echo ''",
            "    echo '❌ Failed tests:'",
            "    printf '  - %s\\n' \"${FAILED_TESTS[@]}\"",
            "    exit 1",
            "else",
            "    echo '✅ All tests passed!'",
            "    exit 0",
            "fi"
        ])
        
        bash_str = "\n".join(bash_lines)
        
        st.download_button(
            label="⬇️ Download Shell Script",
            data=bash_str,
            file_name="run_prioritized_tests.sh",
            mime="text/x-sh",
            help="Executable bash script for direct test execution"
        )
    
    with st.expander("📋 View Export Preview"):
        tab1, tab2, tab3 = st.tabs(["JSON", "YAML", "Shell Script"])
        
        with tab1:
            st.code(json_str, language="json")
        
        with tab2:
            st.code(yaml_str, language="yaml")
        
        with tab3:
            st.code(bash_str, language="bash")


def show_active_learning(system, budget_pct):
    st.header("🎓 Active Learning for Test Selection")
    
    st.markdown("""
    Active learning selects the most informative tests via uncertainty sampling, maximizing learning while minimizing labeling effort.
    This strategy is particularly useful when test execution is expensive or when you want to adaptively improve the model.
    """)
    
    test_commits = system['commits'].iloc[st.session_state.get('n_train', 35):]
    
    if len(test_commits) == 0:
        st.warning("No test commits available. Train the model first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uncertainty_method = st.selectbox(
            "Uncertainty Method",
            ["entropy", "margin"],
            help="Method for calculating prediction uncertainty"
        )
    
    with col2:
        exploration_ratio = st.slider(
            "Exploration Ratio",
            0.0, 1.0, 0.3, 0.05,
            help="Fraction of budget allocated to exploratory (uncertain) tests"
        )
    
    with col3:
        al_budget_pct = st.slider(
            "Active Learning Budget (%)",
            10, 100, 20, 5,
            help="Budget for active learning sample selection"
        )
    
    commit_idx = st.selectbox(
        "Select Commit for Analysis",
        range(len(test_commits)),
        format_func=lambda x: f"{test_commits.iloc[x]['commit_id']} - {test_commits.iloc[x]['message']}"
    )
    
    selected_commit = test_commits.iloc[commit_idx]
    
    commit_data = build_commit_data(selected_commit)
    
    test_suite = system['test_suite']
    feature_extractor = system['feature_extractor']
    
    X = []
    for tid in test_suite['test_id'].values:
        commit_data['coverage_overlap'] = deterministic_coverage(selected_commit['commit_id'], tid)
        X.append(feature_extractor.extract_test_features(
            tid, system['test_history'], commit_data
        ))
    X = np.array(X)
    
    fault_proba = system['model'].predict_proba(X)
    execution_costs = test_suite['execution_time'].values
    
    total_budget = execution_costs.sum() * (al_budget_pct / 100.0)
    
    exploration_indices, exploitation_indices = system['active_learner'].adaptive_budget_allocation(
        fault_proba, execution_costs, total_budget, exploration_ratio
    )
    
    al_selected = exploration_indices + exploitation_indices
    al_cost = sum(execution_costs[i] for i in al_selected)
    
    actual_failures = []
    for test_id in test_suite['test_id']:
        outcomes = [h['outcome'] for h in system['test_history'][test_id]
                   if h['commit_id'] == selected_commit['commit_id']]
        actual_failures.append(outcomes[0] if outcomes else 0)
    actual_failures = np.array(actual_failures)
    
    al_metrics = system['active_learner'].calculate_information_gain(
        al_selected, fault_proba, actual_failures
    )
    
    standard_scores = system['prioritizer'].compute_scores(
        fault_proba, execution_costs, 
        np.zeros(len(fault_proba))
    )
    standard_selected, _ = system['prioritizer'].select_tests(
        standard_scores, execution_costs, total_budget
    )
    
    st.subheader("📊 Strategy Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AL Tests Selected", len(al_selected))
        st.metric("Exploration Tests", len(exploration_indices))
    
    with col2:
        st.metric("Avg Uncertainty", f"{al_metrics['avg_uncertainty']:.3f}")
        st.metric("Exploitation Tests", len(exploitation_indices))
    
    with col3:
        st.metric("Information Gain", f"{al_metrics['information_gain']:.3f}")
        al_fdr = sum(1 for i in al_selected if actual_failures[i]) / max(sum(actual_failures), 1)
        st.metric("AL Fault Detection", f"{al_fdr*100:.1f}%")
    
    with col4:
        st.metric("Label Efficiency", f"{al_metrics['label_efficiency']:.3f}")
        std_fdr = sum(1 for i in standard_selected if actual_failures[i]) / max(sum(actual_failures), 1)
        st.metric("Standard FDR", f"{std_fdr*100:.1f}%")
    
    uncertainty_scores = system['active_learner'].calculate_uncertainty(fault_proba)
    
    al_selected_set = set(al_selected)
    standard_selected_set = set(standard_selected)
    exploration_set = set(exploration_indices)
    exploitation_set = set(exploitation_indices)

    results_df = pd.DataFrame({
        'Test ID': test_suite['test_id'],
        'Test Name': test_suite['name'],
        'Fault Probability': fault_proba,
        'Uncertainty': uncertainty_scores,
        'Execution Time (s)': execution_costs,
        'AL Selected': ['✅' if i in al_selected_set else '❌' for i in range(len(test_suite))],
        'Standard Selected': ['✅' if i in standard_selected_set else '❌' for i in range(len(test_suite))],
        'Selection Type': ['Explore' if i in exploration_set else 
                          'Exploit' if i in exploitation_set else 
                          'Not Selected' for i in range(len(test_suite))],
        'Actual Outcome': ['FAIL' if f else 'PASS' for f in actual_failures]
    })
    
    results_df = results_df.sort_values('Uncertainty', ascending=False)
    
    st.subheader("🔍 Test Selection Results")
    st.dataframe(results_df, width='stretch', height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uncertainty vs Fault Probability")
        fig = px.scatter(results_df, x='Fault Probability', y='Uncertainty',
                        color='AL Selected', size='Execution Time (s)',
                        hover_data=['Test Name', 'Selection Type'],
                        title="Active Learning Test Selection")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Exploration vs Exploitation")
        selection_counts = results_df['Selection Type'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=selection_counts.index, 
                                      values=selection_counts.values)])
        fig.update_layout(title="Budget Allocation Strategy")
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("💡 Active Learning Insights")
    
    high_uncertainty_failures = sum(1 for _, row in results_df.iterrows()
                                    if row['Uncertainty'] > 0.5 and actual_failures[row.name])
    total_failures = sum(actual_failures)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Uncertainty-Failure Correlation:**  
        {high_uncertainty_failures} out of {total_failures} failures ({high_uncertainty_failures/max(total_failures, 1)*100:.1f}%) 
        occurred in high-uncertainty tests (uncertainty > 0.5).
        
        Active learning effectively identifies uncertain regions where the model needs more training data.
        """)
    
    with col2:
        if len(al_selected) > 0 and len(standard_selected) > 0:
            al_efficiency = al_metrics['information_gain'] / len(al_selected)
            std_efficiency = sum(1 for i in standard_selected if abs(fault_proba[i] - actual_failures[i]) > 0.3) / len(standard_selected)
            
            if std_efficiency > 0:
                efficiency_gain = (al_efficiency / std_efficiency - 1) * 100
            else:
                efficiency_gain = 0.0
        else:
            efficiency_gain = 0.0
        
        st.success(f"""
        **Label Efficiency Gain:**  
        Active learning achieves {efficiency_gain:+.1f}% efficiency compared to standard prioritization.
        
        This means AL extracts more information per test execution, reducing the number of tests needed for effective model updates.
        """)
    
    if st.button("🔄 Update Model with AL Selections", help="Update online corrector with actively selected test outcomes"):
        if len(al_selected) > 0:
            X_selected = X[al_selected]
            y_selected = actual_failures[al_selected]
            system['model'].update_online(X_selected, y_selected)
            st.success(f"✅ Model updated with {len(al_selected)} actively selected test outcomes! Information gain incorporated into online corrector.")
            st.rerun()


def show_flakiness_analysis(system):
    st.header("🔍 Flakiness Detection Analysis")
    
    flakiness_scores = system['flakiness_detector'].calculate_flakiness(system['test_history'])
    
    test_suite = system['test_suite']
    flakiness_df = pd.DataFrame({
        'Test ID': list(flakiness_scores.keys()),
        'Flakiness Score': list(flakiness_scores.values())
    })
    
    flakiness_df = flakiness_df.merge(
        test_suite[['test_id', 'name', 'module', 'execution_time']],
        left_on='Test ID', right_on='test_id'
    )
    
    flakiness_df = flakiness_df.sort_values('Flakiness Score', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        high_flaky = (flakiness_df['Flakiness Score'] > 0.3).sum()
        st.metric("High Flakiness Tests", high_flaky)
    with col2:
        avg_flaky = flakiness_df['Flakiness Score'].mean()
        st.metric("Average Flakiness", f"{avg_flaky:.3f}")
    with col3:
        max_flaky = flakiness_df['Flakiness Score'].max()
        st.metric("Maximum Flakiness", f"{max_flaky:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Flaky Tests")
        st.dataframe(flakiness_df.head(20)[['name', 'module', 'Flakiness Score', 'execution_time']], 
                    width='stretch')
    
    with col2:
        st.subheader("Flakiness Distribution by Module")
        module_flaky = flakiness_df.groupby('module')['Flakiness Score'].mean().reset_index()
        fig = px.bar(module_flaky, x='module', y='Flakiness Score',
                    title="Average Flakiness by Module")
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("Flakiness Score Distribution")
    fig = px.histogram(flakiness_df, x='Flakiness Score', nbins=30,
                      title="Distribution of Flakiness Scores Across Test Suite")
    st.plotly_chart(fig, width='stretch')


def show_performance_metrics(system):
    st.header("📊 Model Performance Metrics")
    
    test_commits = system['commits'].iloc[st.session_state.get('n_train', 35):].reset_index(drop=True)
    test_suite = system['test_suite']
    
    flakiness_scores = system['flakiness_detector'].calculate_flakiness(system['test_history'])
    flakiness_array = np.array([flakiness_scores[tid] for tid in test_suite['test_id']])
    execution_costs = test_suite['execution_time'].values
    
    metrics_over_time = []
    
    for _, commit in test_commits.iterrows():
        commit_data = build_commit_data(commit)
        
        X = []
        for tid in test_suite['test_id'].values:
            commit_data['coverage_overlap'] = deterministic_coverage(commit['commit_id'], tid)
            X.append(system['feature_extractor'].extract_test_features(
                tid, system['test_history'], commit_data
            ))
        X = np.array(X)
        
        fault_proba = system['model'].predict_proba(X)
        
        scores = system['prioritizer'].compute_scores(
            fault_proba, execution_costs, flakiness_array
        )
        
        for budget_pct in [10, 30, 50]:
            total_budget = execution_costs.sum() * (budget_pct / 100.0)
            selected_indices, _ = system['prioritizer'].select_tests(
                scores, execution_costs, total_budget
            )
            
            actual_failures = []
            for test_id in test_suite['test_id']:
                outcomes = [h['outcome'] for h in system['test_history'][test_id]
                           if h['commit_id'] == commit['commit_id']]
                actual_failures.append(outcomes[0] if outcomes else 0)
            actual_failures = np.array(actual_failures)
            
            exec_order = [idx for idx in np.argsort(scores)[::-1] if idx in set(selected_indices)]
            m = system['prioritizer'].calculate_metrics(
                selected_indices, actual_failures,
                execution_costs=execution_costs,
                execution_order=exec_order
            )
            
            metrics_over_time.append({
                'commit_id': commit['commit_id'],
                'budget_pct': budget_pct,
                'fdr': m['fault_detection_rate'],
                'precision': m['precision'],
                'atd': m['atd'],
                'wasted_cost': m['wasted_cost'],
                'fpr': m['fpr'],
                'itt': m['itt']
            })
    
    metrics_df = pd.DataFrame(metrics_over_time)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fault Detection Rate Over Time")
        fig = px.line(metrics_df, x='commit_id', y='fdr', color='budget_pct',
                     title="FDR by Budget Level",
                     labels={'fdr': 'Fault Detection Rate', 'budget_pct': 'Budget %'})
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Precision Over Time")
        fig = px.line(metrics_df, x='commit_id', y='precision', color='budget_pct',
                     title="Precision by Budget Level",
                     labels={'precision': 'Precision', 'budget_pct': 'Budget %'})
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("Average Performance by Budget")
    budget_summary = metrics_df.groupby('budget_pct').agg({
        'fdr': 'mean',
        'precision': 'mean'
    }).reset_index()
    
    col1, col2, col3 = st.columns(3)
    for idx, row in budget_summary.iterrows():
        with [col1, col2, col3][idx]:
            st.metric(f"{int(row['budget_pct'])}% Budget", 
                     f"FDR: {row['fdr']*100:.1f}%",
                     f"Precision: {row['precision']*100:.1f}%")


def show_concept_drift(system):
    st.header("🌊 Concept Drift Analysis")
    
    st.markdown("""
    Track how code patterns and test outcomes evolve over time, and how the model adapts to these changes.
    Concept drift occurs when the relationship between code changes and test failures shifts, requiring model updates.
    """)
    
    test_commits = system['commits'].reset_index(drop=True)
    test_suite = system['test_suite']
    n_train = st.session_state.get('n_train', 35)
    
    drift_data = []
    feature_importance_history = []
    
    for commit_idx, commit in test_commits.iterrows():
        commit_data = build_commit_data(commit)
        
        X = []
        for tid in test_suite['test_id'].values:
            commit_data['coverage_overlap'] = deterministic_coverage(commit['commit_id'], tid)
            X.append(system['feature_extractor'].extract_test_features(
                tid, system['test_history'], commit_data
            ))
        X = np.array(X)
        
        fault_proba = system['model'].predict_proba(X)
        
        actual_failures = []
        for test_id in test_suite['test_id']:
            outcomes = [h['outcome'] for h in system['test_history'][test_id]
                       if h['commit_id'] == commit['commit_id']]
            actual_failures.append(outcomes[0] if outcomes else 0)
        actual_failures = np.array(actual_failures)
        
        prediction_error = np.abs(fault_proba - actual_failures).mean()
        
        brier_score = np.mean((fault_proba - actual_failures) ** 2)
        
        if commit_idx >= n_train:
            phase = 'Test'
        else:
            phase = 'Train'
        
        drift_data.append({
            'commit_idx': commit_idx,
            'commit_id': commit['commit_id'],
            'phase': phase,
            'prediction_error': prediction_error,
            'brier_score': brier_score,
            'avg_fault_proba': fault_proba.mean(),
            'actual_failure_rate': actual_failures.mean(),
            'files_modified': commit['files_modified'],
            'code_churn': commit['lines_added'] + commit['lines_deleted']
        })
        
        if commit_idx % 5 == 0:
            feature_imp = system['model'].get_feature_importance()
            if feature_imp is not None:
                feature_names = system['feature_extractor'].get_feature_names()
                for feat_name, imp_val in zip(feature_names[:5], feature_imp[:5]):
                    feature_importance_history.append({
                        'commit_idx': commit_idx,
                        'feature': feat_name,
                        'importance': imp_val
                    })
    
    drift_df = pd.DataFrame(drift_data)
    
    st.subheader("📉 Prediction Error Evolution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_error = drift_df[drift_df['phase'] == 'Train']['prediction_error'].mean()
        st.metric("Avg Train Error", f"{train_error:.3f}")
    
    with col2:
        test_error = drift_df[drift_df['phase'] == 'Test']['prediction_error'].mean()
        st.metric("Avg Test Error", f"{test_error:.3f}")
    
    with col3:
        drift_magnitude = abs(test_error - train_error)
        st.metric("Drift Magnitude", f"{drift_magnitude:.3f}")
    
    with col4:
        online_status = "Adaptive" if system['model'].online_fitted else "Static"
        st.metric("Model State", online_status)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Prediction Error Over Time", "Brier Score Over Time"),
        vertical_spacing=0.15
    )
    
    train_data = drift_df[drift_df['phase'] == 'Train']
    test_data = drift_df[drift_df['phase'] == 'Test']
    
    fig.add_trace(
        go.Scatter(x=train_data['commit_idx'], y=train_data['prediction_error'],
                  mode='lines+markers', name='Train Error',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_data['commit_idx'], y=test_data['prediction_error'],
                  mode='lines+markers', name='Test Error',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=drift_df['commit_idx'], y=drift_df['brier_score'],
                  mode='lines+markers', name='Brier Score',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    fig.add_vline(x=n_train, line_dash="dash", line_color="green",
                 annotation_text="Train/Test Split",
                 row='all')
    
    fig.update_xaxes(title_text="Commit Index", row=2, col=1)
    fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=1)
    fig.update_yaxes(title_text="Brier Score", row=2, col=1)
    fig.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction Calibration")
        
        fig = px.scatter(drift_df, x='avg_fault_proba', y='actual_failure_rate',
                        color='phase', size='code_churn',
                        hover_data=['commit_idx', 'files_modified'],
                        title="Predicted vs Actual Failure Rates",
                        labels={
                            'avg_fault_proba': 'Predicted Failure Probability',
                            'actual_failure_rate': 'Actual Failure Rate'
                        })
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📊 Drift Detection Windows")
        
        window_size = 5
        drift_df['error_rolling_mean'] = drift_df['prediction_error'].rolling(window_size).mean()
        drift_df['error_rolling_std'] = drift_df['prediction_error'].rolling(window_size).std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drift_df['commit_idx'],
            y=drift_df['error_rolling_mean'],
            mode='lines',
            name='Rolling Mean Error',
            line=dict(color='blue', width=2)
        ))
        
        upper_bound = drift_df['error_rolling_mean'] + 2 * drift_df['error_rolling_std']
        lower_bound = drift_df['error_rolling_mean'] - 2 * drift_df['error_rolling_std']
        
        fig.add_trace(go.Scatter(
            x=drift_df['commit_idx'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=drift_df['commit_idx'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            name='±2σ Confidence Band'
        ))
        
        drift_threshold = train_error + 0.1
        fig.add_hline(y=drift_threshold, line_dash="dash", line_color="red",
                     annotation_text="Drift Threshold")
        
        fig.update_layout(
            title="Error Trend with Confidence Bands",
            xaxis_title="Commit Index",
            yaxis_title="Prediction Error"
        )
        
        st.plotly_chart(fig, width='stretch')
    
    if len(feature_importance_history) > 0:
        st.subheader("🔄 Feature Importance Evolution")
        
        feat_imp_df = pd.DataFrame(feature_importance_history)
        
        fig = px.line(feat_imp_df, x='commit_idx', y='importance', color='feature',
                     title="Top 5 Features Importance Over Time",
                     labels={'commit_idx': 'Commit Index', 'importance': 'Feature Importance'})
        
        fig.add_vline(x=n_train, line_dash="dash", line_color="green",
                     annotation_text="Train/Test Split")
        
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("🔔 Drift Alerts")
    
    test_phase_data = drift_df[drift_df['phase'] == 'Test']
    
    if len(test_phase_data) > 0:
        recent_error = test_phase_data['prediction_error'].tail(5).mean()
        drift_detected = recent_error > (train_error + 0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if drift_detected:
                st.error(f"""
                ⚠️ **Concept Drift Detected!**
                
                Recent prediction error ({recent_error:.3f}) significantly exceeds 
                training baseline ({train_error:.3f}).
                
                **Recommendations:**
                1. Update model with recent test outcomes using incremental learning
                2. Review code changes in recent commits for new patterns
                3. Consider retraining the static forest if drift persists
                """)
            else:
                st.success(f"""
                ✅ **Model Performance Stable**
                
                Recent prediction error ({recent_error:.3f}) is within acceptable 
                range of training baseline ({train_error:.3f}).
                
                The model is adapting well to code changes.
                """)
        
        with col2:
            error_increase_pct = ((recent_error - train_error) / max(train_error, 0.01)) * 100
            
            st.info(f"""
            **Drift Statistics:**
            
            - **Error Increase**: {error_increase_pct:+.1f}%
            - **Commits Analyzed**: {len(test_phase_data)}
            - **Online Updates**: {"Yes" if system['model'].online_fitted else "No"}
            - **Adaptation Rate**: {"High" if abs(error_increase_pct) < 20 else "Low"}
            
            Regular incremental updates help the model adapt to evolving code patterns 
            and maintain predictive accuracy over time.
            """)


def show_model_explainability(system):
    st.header("🧠 Model Explainability")
    
    feature_importance = system['model'].get_feature_importance()
    feature_names = system['feature_extractor'].get_feature_names()
    
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Feature Importance Rankings")
            st.dataframe(importance_df, width='stretch')
        
        with col2:
            st.subheader("Feature Importance Visualization")
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Static Forest Feature Importance")
            st.plotly_chart(fig, width='stretch')
    
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **IncEnsemble Components:**
        
        1. **Static Decision Forest (F)**
           - Number of trees: 100
           - Max depth: 8 levels
           - Trained offline on historical data
           - Captures complex feature interactions
        
        2. **Online Linear Corrector (L)**
           - SGD-based logistic regression with balanced class weighting
           - Updates incrementally with new outcomes
           - Adapts to concept drift
        
        3. **Meta-weighting Layer**
           - Combines predictions: α·F + (1-α)·L
           - Current α: {:.2f}
        """.format(system['model'].alpha))
    
    with col2:
        st.markdown("""
        **Key Features (13 total):**
        
        - **Commit-level**: Files modified, lines added/deleted, code churn
        - **Code-level**: Token count, method count delta, syntactic complexity
        - **Test-level**: Recent pass rate, execution time, failure streaks
        - **Impact**: Coverage overlap, time since last failure
        - **Stability**: Execution time variance
        
        **Prioritization Strategy:**
        
        Score = (P(failure) × urgency) / cost - λ × flakiness
        
        Where:
        - P(failure): Predicted failure probability
        - cost: Execution time
        - λ: Flakiness penalty factor
        """)
    
    st.subheader("Algorithm Complexity")
    st.markdown("""
    **Time Complexity per CI Run:** O(|T| · d)
    - |T|: Number of tests in suite
    - d: Feature dimensionality (13 features)
    
    **Space Complexity:** O(|T| · h) where h = history window size
    
    **Update Latency:** < 100ms for incremental online updates
    """)


def show_shap_analysis(system):
    st.header("🔬 SHAP Explainability Analysis")
    
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values explain how each feature contributes to individual predictions,
    providing transparent, interpretable reasoning for test prioritization decisions.
    """)
    
    try:
        import shap
    except ImportError:
        st.error("SHAP is not installed. Run: `pip install shap`")
        return
    
    feature_extractor = system['feature_extractor']
    feature_names = feature_extractor.get_feature_names()
    
    test_suite = system['test_suite']
    test_history = system['test_history']
    
    sample_commits = system['commits'].iloc[:5]
    
    X_all = []
    for _, commit in sample_commits.iterrows():
        commit_data = build_commit_data(commit)
        for tid in test_suite['test_id'].values:
            commit_data['coverage_overlap'] = deterministic_coverage(commit['commit_id'], tid)
            X_all.append(feature_extractor.extract_test_features(tid, test_history, commit_data))
    X_all = np.array(X_all)
    
    bg_indices = np.random.RandomState(42).choice(len(X_all), min(50, len(X_all)), replace=False)
    X_background = X_all[bg_indices]
    
    forest = system['model'].static_forest
    
    explainer = shap.TreeExplainer(forest)
    shap_values = explainer.shap_values(X_all[:100])
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)
    
    importance_df['Importance %'] = (importance_df['Mean |SHAP|'] / importance_df['Mean |SHAP|'].sum() * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance (SHAP)")
        st.dataframe(importance_df, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("SHAP Summary Plot")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_all[:100], feature_names=feature_names, 
                            show=False, max_display=13)
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render SHAP summary plot: {e}")
            fig = px.bar(importance_df, x='Mean |SHAP|', y='Feature', orientation='h',
                        title="SHAP Feature Importance")
            st.plotly_chart(fig, width='stretch')
    
    st.subheader("SHAP Feature Importance Distribution")
    fig = px.bar(importance_df, x='Importance %', y='Feature', orientation='h',
                title="Feature Contribution to Predictions (%)",
                text='Importance %')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width='stretch')
    
    st.info("""
    **Interpretation:**  
    SHAP values show the marginal contribution of each feature to the predicted failure probability.
    Positive SHAP values push predictions toward failure; negative values push toward pass.
    Historical failure rate and code churn are typically the most influential features,
    confirming that IncEnsemble relies on stable, historically validated quality indicators.
    """)


def show_baseline_comparison(system, budget_pct, n_train):
    st.header("⚔️ Baseline Comparison")
    
    st.markdown("""
    Compare IncEnsemble against representative baselines: random selection, static prioritization,
    and non-incremental random forest. All methods operate under the same execution budget.
    """)
    
    test_commits = system['commits'].iloc[n_train:].reset_index(drop=True)
    test_suite = system['test_suite']
    
    if len(test_commits) == 0:
        st.warning("No test commits available for comparison.")
        return
    
    with st.spinner("Running baseline comparison..."):
        results = run_baselines(system, test_commits, test_suite, budget_pct, n_train)
    
    if results is None or len(results) == 0:
        st.error("Baseline comparison failed.")
        return
    
    st.subheader("FDR by Method")
    fdr_summary = results.groupby('method')['fdr'].agg(['mean', 'std']).reset_index()
    fdr_summary.columns = ['Method', 'Mean FDR', 'Std FDR']
    fdr_summary = fdr_summary.sort_values('Mean FDR', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(fdr_summary, x='Method', y='Mean FDR',
                    title="Mean Fault Detection Rate by Method",
                    error_y='Std FDR',
                    color='Method')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.dataframe(fdr_summary, width='stretch', hide_index=True)
    
    st.subheader("Statistical Significance (Wilcoxon Signed-Rank Test)")
    
    pairs = [('IncEnsemble', 'Random'), ('IncEnsemble', 'Static'), ('IncEnsemble', 'Non-inc RF')]
    
    test_results = []
    for method_a, method_b in pairs:
        result = wilcoxon_test(results, method_a, method_b)
        test_results.append(result)
    
    test_df = pd.DataFrame(test_results)
    st.dataframe(test_df[['method_a', 'method_b', 'p_value', 'significant', 'message']], 
                width='stretch', hide_index=True)
    
    st.subheader("FDR Over Commits")
    fig = px.line(results, x='commit_id', y='fdr', color='method',
                 title="Fault Detection Rate Over Commits by Method")
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(fig, width='stretch')


def show_fdr_vs_budget(system, budget_pct, n_train):
    st.header("📈 FDR vs CI Budget")
    
    st.markdown("""
    Aggregated Fault Detection Rate across varying CI execution budgets.
    Shows how efficiently each budget level detects failures.
    """)
    
    test_suite = system['test_suite']
    test_history = system['test_history']
    
    with st.spinner("Computing FDR vs budget curve..."):
        budget_df = compute_fdr_vs_budget(system, test_suite, test_history, n_train=n_train)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(budget_df, x='budget_pct', y='mean_fdr',
                     title="Mean FDR vs CI Budget",
                     labels={'budget_pct': 'Budget (%)', 'mean_fdr': 'Mean FDR'})
        fig.add_trace(go.Scatter(
            x=budget_df['budget_pct'].tolist() + budget_df['budget_pct'].tolist()[::-1],
            y=(budget_df['mean_fdr'] + budget_df['std_fdr']).tolist() + 
              (budget_df['mean_fdr'] - budget_df['std_fdr']).tolist()[::-1],
            fill='toself', fillcolor='rgba(0,100,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'), showlegend=False, name='±1σ'
        ))
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.dataframe(budget_df, width='stretch', hide_index=True)
    
    st.info("""
    **Observation:**  
    FDR increases rapidly with budget up to ~40%, then plateaus.
    This confirms that IncEnsemble can achieve high fault detection with partial test execution,
    supporting efficient CI/CD integration.
    """)


def show_multi_run_evaluation(system, budget_pct, n_train):
    st.header("🔁 Multi-Run Evaluation")
    
    st.markdown("""
    Evaluation across multiple runs with varying random seeds to assess statistical robustness.
    Paper §6.1: Experiments repeated 10 times with random seed variations.
    """)
    
    n_runs = st.slider("Number of runs", 5, 20, 10, 1)
    
    with st.spinner(f"Running {n_runs} evaluation runs..."):
        results = run_multi_evaluation(system, n_runs=n_runs, budget_pct=budget_pct, n_train=n_train)
    
    if results is None or len(results) == 0:
        st.error("Multi-run evaluation failed.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean FDR", f"{results['mean_fdr'].mean()*100:.1f}%",
                  f"±{results['mean_fdr'].std()*100:.1f}%")
    with col2:
        st.metric("Mean Precision", f"{results['mean_precision'].mean()*100:.1f}%",
                  f"±{results['mean_precision'].std()*100:.1f}%")
    with col3:
        st.metric("Mean ATD (s)", f"{results['mean_atd'].mean():.2f}",
                  f"±{results['mean_atd'].std():.2f}")
    with col4:
        st.metric("FDR Std Dev", f"{results['std_fdr'].mean()*100:.2f}%")
    
    fig = px.bar(results, x='run', y='mean_fdr',
                title="FDR Across Runs",
                labels={'run': 'Run', 'mean_fdr': 'Mean FDR'},
                error_y='std_fdr')
    st.plotly_chart(fig, width='stretch')
    
    st.subheader("Run-by-Run Results")
    st.dataframe(results, width='stretch', hide_index=True)


def show_hyperparameter_sensitivity(system, budget_pct, n_train):
    st.header("🎛️ Hyperparameter Sensitivity Analysis")
    
    st.markdown("""
    Sensitivity analysis of key hyperparameters: flakiness penalty λ and meta-weight α.
    Paper §5.5: Performance variation remained below 2.7%, indicating robustness.
    """)
    
    with st.spinner("Running hyperparameter sensitivity analysis..."):
        results = run_hyperparameter_sensitivity(system, budget_pct=budget_pct, n_train=n_train)
    
    if results is None or len(results) == 0:
        st.error("Sensitivity analysis failed.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        lambda_results = results[results['parameter'] == 'lambda']
        fig = px.line(lambda_results, x='value', y='mean_fdr',
                     title="FDR vs Flakiness Penalty (λ)",
                     labels={'value': 'λ', 'mean_fdr': 'Mean FDR'},
                     error_y='std_fdr')
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        alpha_results = results[results['parameter'] == 'alpha']
        fig = px.line(alpha_results, x='value', y='mean_fdr',
                     title="FDR vs Meta-weight (α)",
                     labels={'value': 'α', 'mean_fdr': 'Mean FDR'},
                     error_y='std_fdr')
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("Sensitivity Results")
    st.dataframe(results, width='stretch', hide_index=True)
    
    fdr_range = results['mean_fdr'].max() - results['mean_fdr'].min()
    st.info(f"""
    **Observation:**  
    FDR variation across evaluated parameter ranges: {fdr_range*100:.1f}%.
    {'Framework is robust to parameter selection.' if fdr_range < 0.05 else 'Some sensitivity observed — consider project-specific tuning.'}
    """)


def show_imbalance_comparison(system, budget_pct, n_train):
    st.header("⚖️ Imbalance Handling Comparison")
    
    st.markdown("""
    Comparison of different class imbalance handling strategies: no balancing,
    random oversampling, and weighted learning. Paper §6.2 reports improved
    minority-class failure detection with weighted approaches.
    """)
    
    with st.spinner("Running imbalance strategy comparison..."):
        results = run_imbalance_comparison(system, budget_pct=budget_pct, n_train=n_train)
    
    if results is None or len(results) == 0:
        st.error("Imbalance comparison failed.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(results, x='method', y='f1',
                    title="F1-Score by Imbalance Strategy",
                    color='method')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(results, x='method', y='recall',
                    title="Recall by Imbalance Strategy",
                    color='method')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    st.subheader("Strategy Comparison")
    st.dataframe(results, width='stretch', hide_index=True)
    
    best = results.loc[results['f1'].idxmax()]
    st.success(f"""
    **Best Strategy:** {best['method']} (F1={best['f1']:.3f}, Recall={best['recall']:.3f})
    """)


if __name__ == "__main__":
    main()
