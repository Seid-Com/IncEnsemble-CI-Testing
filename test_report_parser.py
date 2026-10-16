import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class JUnitParser:
    def parse_junit_xml(self, xml_path: str, file_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        test_cases = []
        execution_logs = []
        
        testsuites = root.findall('.//testsuite')
        if not testsuites:
            testsuites = [root] if root.tag == 'testsuite' else []
        
        if file_id is None:
            file_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        
        commit_id = f"commit_{file_id}"
        
        for testsuite in testsuites:
            suite_name = testsuite.get('name', 'unknown_suite')
            
            for testcase in testsuite.findall('testcase'):
                test_name = testcase.get('name', 'unnamed_test')
                classname = testcase.get('classname', suite_name)
                time_str = testcase.get('time', '0.0')
                
                try:
                    execution_time = float(time_str)
                except ValueError:
                    execution_time = 0.0
                
                failure = testcase.find('failure')
                error = testcase.find('error')
                skipped = testcase.find('skipped')
                
                if failure is not None or error is not None:
                    outcome = 1
                    status = 'failed'
                elif skipped is not None:
                    outcome = 0
                    status = 'skipped'
                else:
                    outcome = 0
                    status = 'passed'
                
                test_id = f"{classname}.{test_name}_{file_id}"
                
                test_cases.append({
                    'test_id': test_id,
                    'name': test_name,
                    'module': classname,
                    'execution_time': execution_time,
                    'category': 'unit'
                })
                
                execution_logs.append({
                    'test_id': test_id,
                    'commit_id': commit_id,
                    'outcome': outcome,
                    'status': status,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                })
        
        test_suite_df = pd.DataFrame(test_cases)
        execution_logs_df = pd.DataFrame(execution_logs)
        
        commits_df = pd.DataFrame([{
            'commit_id': commit_id,
            'message': 'Imported from JUnit XML',
            'author': 'imported_user',
            'timestamp': datetime.now().isoformat(),
            'files_modified': 5,
            'lines_added': 50,
            'lines_deleted': 20
        }])
        
        return test_suite_df, commits_df, execution_logs_df


class TestNGParser:
    def parse_testng_xml(self, xml_path: str, file_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        test_cases = []
        execution_logs = []
        
        if file_id is None:
            file_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        
        commit_id = f"commit_{file_id}"
        
        for suite in root.findall('.//suite'):
            suite_name = suite.get('name', 'unknown_suite')
            
            for test in suite.findall('.//test'):
                test_name_attr = test.get('name', 'unknown_test')
                
                for class_elem in test.findall('.//class'):
                    classname = class_elem.get('name', suite_name)
                    
                    for test_method in class_elem.findall('.//test-method'):
                        method_name = test_method.get('name', 'unnamed_test')
                        status_attr = test_method.get('status', 'PASS')
                        duration_ms = test_method.get('duration-ms', '0')
                        
                        try:
                            execution_time = float(duration_ms) / 1000.0
                        except ValueError:
                            execution_time = 0.0
                        
                        if status_attr == 'PASS':
                            outcome = 0
                            status = 'passed'
                        elif status_attr == 'FAIL':
                            outcome = 1
                            status = 'failed'
                        elif status_attr == 'SKIP':
                            outcome = 0
                            status = 'skipped'
                        else:
                            outcome = 0
                            status = 'passed'
                        
                        test_id = f"{classname}.{method_name}_{file_id}"
                        
                        test_cases.append({
                            'test_id': test_id,
                            'name': method_name,
                            'module': classname,
                            'execution_time': execution_time,
                            'category': 'unit'
                        })
                        
                        execution_logs.append({
                            'test_id': test_id,
                            'commit_id': commit_id,
                            'outcome': outcome,
                            'status': status,
                            'execution_time': execution_time,
                            'timestamp': datetime.now().isoformat()
                        })
        
        test_suite_df = pd.DataFrame(test_cases)
        execution_logs_df = pd.DataFrame(execution_logs)
        
        commits_df = pd.DataFrame([{
            'commit_id': commit_id,
            'message': 'Imported from TestNG XML',
            'author': 'imported_user',
            'timestamp': datetime.now().isoformat(),
            'files_modified': 5,
            'lines_added': 50,
            'lines_deleted': 20
        }])
        
        return test_suite_df, commits_df, execution_logs_df


class TestReportParser:
    def __init__(self):
        self.junit_parser = JUnitParser()
        self.testng_parser = TestNGParser()
    
    def detect_format(self, xml_path: str) -> str:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            if root.tag in ['testsuites', 'testsuite']:
                return 'junit'
            elif root.tag in ['suite', 'testng-results']:
                return 'testng'
            else:
                return 'unknown'
        except Exception as e:
            return 'unknown'
    
    def parse_report(self, xml_path: str, format_type: str = None, file_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if format_type is None:
            format_type = self.detect_format(xml_path)
        
        if format_type == 'junit':
            return self.junit_parser.parse_junit_xml(xml_path, file_id)
        elif format_type == 'testng':
            return self.testng_parser.parse_testng_xml(xml_path, file_id)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def merge_multiple_reports(self, reports: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_test_suites = []
        all_commits = []
        all_execution_logs = []
        
        for test_suite, commits, execution_logs in reports:
            all_test_suites.append(test_suite)
            all_commits.append(commits)
            all_execution_logs.append(execution_logs)
        
        merged_test_suite = pd.concat(all_test_suites, ignore_index=True)
        merged_commits = pd.concat(all_commits, ignore_index=True)
        merged_execution_logs = pd.concat(all_execution_logs, ignore_index=True)
        
        return merged_test_suite, merged_commits, merged_execution_logs
