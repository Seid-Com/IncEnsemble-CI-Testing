import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class IncEnsemble:
    def __init__(self, n_trees=100, max_depth=8, alpha=0.6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.alpha = alpha
        
        self.static_forest = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.online_corrector = SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            warm_start=True
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.online_fitted = False
        self._seen_n0 = 0
        self._seen_n1 = 0
        self.pi_train = 0.15
        self._prior_correction = 0.0
        
    def fit_static(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.static_forest.fit(X_scaled, y)
        self.is_fitted = True
        self.pi_train = float(np.mean(y))
        return self
        
    def update_online(self, X, y):
        if len(X) == 0:
            return self

        X_scaled = self.scaler.transform(X)
        y = np.asarray(y)

        # Class weights from cumulative online counts (more stable than
        # per-batch balancing, which miscalibrates probabilities).
        n1_batch = int((y == 1).sum())
        n0_batch = len(y) - n1_batch
        self._seen_n1 += n1_batch
        self._seen_n0 += n0_batch
        total = max(self._seen_n0 + self._seen_n1, 1)
        # Prior-shift correction: online labels are drawn from the selected
        # (failure-enriched) subset, so correct the online prior back to the
        # full-population training prior.
        pi_obs = (self._seen_n1 + 1.0) / (total + 2.0)
        pi_obs = min(max(pi_obs, 1e-4), 1 - 1e-4)
        pi_tr = min(max(self.pi_train, 1e-4), 1 - 1e-4)
        self._prior_correction = (
            np.log(pi_tr / (1 - pi_tr)) - np.log(pi_obs / (1 - pi_obs)))
        sample_weight = np.where(
            y == 1,
            total / (2.0 * max(self._seen_n1, 1)),
            total / (2.0 * max(self._seen_n0, 1)))
        sample_weight = np.clip(sample_weight, 0.2, 5.0)

        if not self.online_fitted:
            self.online_corrector.partial_fit(
                X_scaled, y, classes=[0, 1], sample_weight=sample_weight)
            self.online_fitted = True
        else:
            self.online_corrector.partial_fit(
                X_scaled, y, sample_weight=sample_weight)

        return self
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_static first.")
        
        X_scaled = self.scaler.transform(X)
        
        forest_proba = self.static_forest.predict_proba(X_scaled)[:, 1]
        
        if self.online_fitted:
            online_proba = self.online_corrector.predict_proba(X_scaled)[:, 1]
            if self._prior_correction != 0.0:
                eps = 1e-6
                logits = np.log(np.clip(online_proba, eps, 1 - eps) /
                                np.clip(1 - online_proba, eps, 1 - eps))
                online_proba = 1.0 / (1.0 + np.exp(-(logits + self._prior_correction)))
            combined_proba = self.alpha * forest_proba + (1 - self.alpha) * online_proba
        else:
            combined_proba = forest_proba
        
        return combined_proba
    
    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        return self.static_forest.feature_importances_
    
    def save(self, filepath):
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        return joblib.load(filepath)


class FlakinessDetector:
    def __init__(self, window_size=20, transition_weight=0.7, variance_weight=0.3):
        self.window_size = window_size
        self.transition_weight = transition_weight
        self.variance_weight = variance_weight
        
    def calculate_flakiness(self, test_history):
        flakiness_scores = {}
        
        for test_id, history_records in test_history.items():
            if len(history_records) < 2:
                flakiness_scores[test_id] = 0.0
                continue
            
            recent_records = history_records[-self.window_size:]
            
            if len(recent_records) == 0:
                flakiness_scores[test_id] = 0.0
                continue
            
            outcome_values = [record['outcome'] for record in recent_records]
            
            transitions = sum(1 for i in range(len(outcome_values)-1) 
                            if outcome_values[i] != outcome_values[i+1])
            
            variance = np.var(outcome_values)
            
            flakiness = ((transitions / max(len(outcome_values)-1, 1)) * self.transition_weight 
                        + variance * self.variance_weight)
            
            flakiness_scores[test_id] = min(flakiness, 1.0)
        
        return flakiness_scores


class TestPrioritizer:
    def __init__(self, lambda_penalty=0.3):
        self.lambda_penalty = lambda_penalty
        
    def compute_scores(self, fault_proba, execution_costs, flakiness_scores, urgency=None):
        n_tests = len(fault_proba)

        if urgency is None:
            urgency = np.ones(n_tests)

        scores = (fault_proba * urgency) / np.maximum(execution_costs, 0.1)

        # Scale-invariant flakiness discount: a multiplicative term keeps the
        # penalty commensurate with the cost-normalised score magnitude.
        scores = scores * (1.0 - self.lambda_penalty * np.clip(flakiness_scores, 0.0, 1.0))

        return scores
    
    def select_tests(self, scores, execution_costs, budget):
        ranked_indices = np.argsort(scores)[::-1]
        
        selected = []
        total_cost = 0
        
        for idx in ranked_indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(idx)
                total_cost += execution_costs[idx]
            
            if total_cost >= budget:
                break
        
        return selected, total_cost
    
    def calculate_metrics(self, selected_indices, actual_failures, execution_costs=None, execution_order=None):
        if len(actual_failures) == 0:
            return {
                'fault_detection_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'fpr': 0.0,
                'atd': 0.0,
                'wasted_cost': 0.0,
                'mul': 0.0,
                'itt': 0.0
            }
        
        selected_set = set(selected_indices)
        detected = sum(1 for idx in selected_indices if actual_failures[idx])
        total_failures = sum(actual_failures)
        
        fdr = detected / max(total_failures, 1)
        precision = detected / max(len(selected_indices), 1)
        recall = fdr
        
        false_positives = sum(1 for idx in selected_indices if not actual_failures[idx])
        true_negatives = sum(1 for i in range(len(actual_failures)) 
                            if i not in selected_set and not actual_failures[i])
        fpr = false_positives / max(false_positives + true_negatives, 1)
        
        atd = 0.0
        wasted_cost = 0.0
        if execution_costs is not None and execution_order is not None:
            cum_cost = 0.0
            first_fail_time = None
            for idx in execution_order:
                cum_cost += execution_costs[idx]
                if actual_failures[idx] and first_fail_time is None:
                    first_fail_time = cum_cost
                if idx in selected_set and not actual_failures[idx]:
                    wasted_cost += execution_costs[idx]
            atd = first_fail_time if first_fail_time is not None else cum_cost
        
        mul = 0.005
        if execution_costs is not None and len(selected_indices) > 0:
            selected_costs = execution_costs[list(selected_indices)]
            not_selected_mask = np.ones(len(execution_costs), dtype=bool)
            not_selected_mask[list(selected_indices)] = False
            if not_selected_mask.any():
                not_selected_costs = execution_costs[not_selected_mask]
                not_selected_failures = actual_failures[not_selected_mask]
                if not_selected_failures.sum() > 0:
                    missed_weight = np.sum(not_selected_costs * not_selected_failures)
                    total_weight = np.sum(execution_costs * actual_failures)
                    mul = missed_weight / max(total_weight, 1e-10)
        
        itt = 0.0
        if execution_costs is not None and len(selected_indices) > 0:
            itt = np.mean([execution_costs[i] for i in selected_indices])
        
        return {
            'fault_detection_rate': fdr,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'atd': atd,
            'wasted_cost': wasted_cost,
            'mul': mul,
            'itt': itt,
            'detected': detected,
            'total_failures': total_failures
        }
