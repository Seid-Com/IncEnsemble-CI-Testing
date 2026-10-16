import numpy as np
from typing import List, Tuple


class ActiveLearner:
    def __init__(self, utility_weight=1.0, cost_weight=1.0):
        self.utility_weight = utility_weight
        self.cost_weight = cost_weight
        
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        entropy = -(probabilities * np.log2(probabilities + 1e-10) + 
                   (1 - probabilities) * np.log2(1 - probabilities + 1e-10))
        
        return entropy
    
    def calculate_margin(self, probabilities: np.ndarray) -> np.ndarray:
        margin = np.abs(probabilities - 0.5)
        
        uncertainty = 1.0 - margin
        
        return uncertainty
    
    def calculate_expected_utility(self, 
                                   probabilities: np.ndarray,
                                   execution_costs: np.ndarray,
                                   uncertainty_method='entropy') -> np.ndarray:
        if uncertainty_method == 'entropy':
            uncertainty = self.calculate_uncertainty(probabilities)
        elif uncertainty_method == 'margin':
            uncertainty = self.calculate_margin(probabilities)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
        expected_utility = (self.utility_weight * uncertainty) / (
            self.cost_weight * np.maximum(execution_costs, 0.1)
        )
        
        return expected_utility
    
    def select_informative_tests(self, 
                                 probabilities: np.ndarray,
                                 execution_costs: np.ndarray,
                                 budget: float,
                                 uncertainty_method='entropy') -> Tuple[List[int], float]:
        expected_utilities = self.calculate_expected_utility(
            probabilities, execution_costs, uncertainty_method
        )
        
        ranked_indices = np.argsort(expected_utilities)[::-1]
        
        selected = []
        total_cost = 0
        
        for idx in ranked_indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(int(idx))
                total_cost += execution_costs[idx]
            
            if total_cost >= budget:
                break
        
        return selected, total_cost
    
    def calculate_information_gain(self, 
                                   selected_indices: List[int],
                                   probabilities: np.ndarray,
                                   actual_outcomes: np.ndarray) -> dict:
        if len(selected_indices) == 0:
            return {
                'avg_uncertainty': 0.0,
                'information_gain': 0.0,
                'label_efficiency': 0.0,
                'n_labels_acquired': 0
            }
        
        selected_probs = probabilities[selected_indices]
        selected_outcomes = actual_outcomes[selected_indices].astype(float)
        
        avg_uncertainty = self.calculate_uncertainty(selected_probs).mean()
        
        prediction_errors = np.abs(selected_probs - selected_outcomes)
        information_gain = prediction_errors.mean()
        
        n_selected = len(selected_indices)
        n_total = len(probabilities)
        selection_ratio = n_selected / max(n_total, 1)
        
        if selection_ratio > 0:
            label_efficiency = information_gain / selection_ratio
        else:
            label_efficiency = 0.0
        
        return {
            'avg_uncertainty': float(avg_uncertainty),
            'information_gain': float(information_gain),
            'label_efficiency': float(label_efficiency),
            'n_labels_acquired': len(selected_indices)
        }
    
    def adaptive_budget_allocation(self,
                                   probabilities: np.ndarray,
                                   execution_costs: np.ndarray,
                                   total_budget: float,
                                   exploration_ratio=0.3) -> Tuple[List[int], List[int]]:
        exploration_budget = total_budget * exploration_ratio
        
        exploration_indices, exploration_cost = self.select_informative_tests(
            probabilities, execution_costs, exploration_budget, uncertainty_method='entropy'
        )
        
        remaining_budget = max(0.0, total_budget - exploration_cost)
        
        remaining_indices = [i for i in range(len(probabilities)) 
                           if i not in exploration_indices]
        
        exploitation_indices = []
        
        if len(remaining_indices) > 0 and remaining_budget > 0:
            remaining_probs = probabilities[remaining_indices]
            
            exploitation_ranked = np.argsort(remaining_probs)[::-1]
            
            exploitation_cost = 0.0
            
            for local_idx in exploitation_ranked:
                global_idx = remaining_indices[local_idx]
                test_cost = execution_costs[global_idx]
                
                if exploitation_cost + test_cost <= remaining_budget:
                    exploitation_indices.append(global_idx)
                    exploitation_cost += test_cost
                
                if exploitation_cost >= remaining_budget:
                    break
        
        return exploration_indices, exploitation_indices
    
    def query_by_committee(self,
                          predictions_list: List[np.ndarray],
                          execution_costs: np.ndarray,
                          budget: float) -> Tuple[List[int], float]:
        if len(predictions_list) < 2:
            raise ValueError("Query by committee requires at least 2 predictors")
        
        predictions_array = np.array(predictions_list)
        
        disagreement = np.std(predictions_array, axis=0)
        
        utility_scores = disagreement / np.maximum(execution_costs, 0.1)
        
        ranked_indices = np.argsort(utility_scores)[::-1]
        
        selected = []
        total_cost = 0
        
        for idx in ranked_indices:
            if total_cost + execution_costs[idx] <= budget:
                selected.append(int(idx))
                total_cost += execution_costs[idx]
            
            if total_cost >= budget:
                break
        
        return selected, total_cost
