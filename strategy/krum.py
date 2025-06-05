from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utilities.utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters, aggregate_ndarrays_weighted


# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]









# Krum Strategy Implementation
class Krum(FedAvg):
    """Krum aggregation strategy implementation.
    
    Based on the paper: https://arxiv.org/pdf/1703.02757
    A Byzantine-robust strategy that selects the most representative client model.
    """
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[EvaluateFnType] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        # Krum-specific parameters
        num_byzantine: int = 0,          # Expected number of Byzantine clients
        multi_krum: bool = False,        # Whether to use Multi-Krum
        krum_m: Optional[int] = None,    # Number of models to select in Multi-Krum
    ) -> None:
        super().__init__(
            initial_parameters=initial_parameters,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        # Krum-specific attributes
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        self.krum_m = krum_m
    
    def _calculate_distance(self, model_a: List[np.ndarray], model_b: List[np.ndarray]) -> float:
        """Calculate the Euclidean distance between two models."""
        distance = 0.0
        for layer_a, layer_b in zip(model_a, model_b):
            # Flatten parameters if needed
            flat_a = layer_a.flatten()
            flat_b = layer_b.flatten()
            # Add squared Euclidean distance for this layer
            distance += np.sum((flat_a - flat_b) ** 2)
        return float(distance)
    
    def _apply_krum(
        self, 
        client_models: List[List[np.ndarray]], 
        client_examples: List[int]
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Apply Krum algorithm to select client model(s)."""
        num_clients = len(client_models)
        
        # Need at least 2f+3 clients for Krum to be well-defined
        min_clients_required = 2 * self.num_byzantine + 3
        if num_clients < min_clients_required:
            print(f"[Server] Warning: Not enough clients for Krum (need {min_clients_required}, have {num_clients}). Using default aggregation.")
            return [], []  # Return empty lists instead of None values
        
        # Calculate pairwise distances between all models
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i+1, num_clients):  # Only calculate upper triangle
                dist = self._calculate_distance(client_models[i], client_models[j])
                distances[i, j] = dist
                distances[j, i] = dist  # Symmetry
        
        # Calculate scores: sum of distances to n-f-2 closest neighbors
        n = num_clients
        f = self.num_byzantine
        num_neighbors = n - f - 2  # Number of closest neighbors to consider
        
        # For each client, get the n-f-2 closest other clients
        scores = np.zeros(num_clients)
        for i in range(num_clients):
            # Get distances to other clients, sort them
            client_distances = distances[i]
            # Set distance to self to infinity to exclude it
            client_distances_no_self = client_distances.copy()
            client_distances_no_self[i] = float('inf')
            # Get indices of closest neighbors
            closest_indices = np.argsort(client_distances_no_self)[:num_neighbors]
            # Sum distances to closest neighbors
            scores[i] = np.sum(client_distances_no_self[closest_indices])
        
        # If using Multi-Krum, select m clients with lowest scores
        if self.multi_krum:
            # Determine number of models to select
            m = self.krum_m if self.krum_m is not None else max(1, n - f)
            m = min(m, n)  # Ensure m doesn't exceed number of clients
            
            # Get indices of clients with lowest scores
            selected_indices = np.argsort(scores)[:m]
            
            # Get selected models and their weights based on example counts
            selected_models = [client_models[i] for i in selected_indices]
            selected_examples = [client_examples[i] for i in selected_indices]
            
            return selected_models, selected_examples
        else:
            # Single Krum: select the client with the lowest score
            best_idx = np.argmin(scores)
            # Return as a single-item list to maintain consistency with other methods
            return [client_models[best_idx]], [client_examples[best_idx]]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Krum."""
        if not results:
            return None, {}

        # Extract weights and client information
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Extract client models and example counts
        client_models = [weights for weights, _ in weights_results]
        client_examples = [num_examples for _, num_examples in weights_results]
        
        # Apply Krum to select model(s)
        selected_models, selected_examples = self._apply_krum(client_models, client_examples)
        
        # If Krum couldn't select a model (e.g., not enough clients), use normal FedAvg
        if not selected_models: # Changed from `selected_models is None`
            return super().aggregate_fit(server_round, results, failures)
        
        # If Multi-Krum, average the selected models
        if self.multi_krum and len(selected_models) > 1:
            # Calculate weights for averaging
            total_examples = sum(selected_examples)
            if total_examples > 0:
                model_weights = [e / total_examples for e in selected_examples]
            else:
                # If no example count info, use uniform weights
                model_weights = [1.0 / len(selected_models)] * len(selected_models)
                
            # Perform weighted averaging of selected models
            parameters_aggregated = ndarrays_to_parameters(
                aggregate_ndarrays_weighted(selected_models, model_weights)
            )
        else:
            # Single Krum: use the selected model directly
            parameters_aggregated = ndarrays_to_parameters(selected_models[0])
        
        # Prepare metrics with standard aggregation
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) 
                          for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            # Add Krum-specific metrics
            metrics_aggregated["krum_selected_clients"] = 1 if not self.multi_krum else len(selected_models)
        
        return parameters_aggregated, metrics_aggregated


# Custom Krum Strategy with metrics printing
class CustomKrum(Krum, BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training."""
        # Get client/config pairs from parent class
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Add round number to each client's configuration
        for _, fit_ins in client_instructions:
            if fit_ins.config is None:
                fit_ins.config = {}
            fit_ins.config["round"] = server_round
        
        return client_instructions
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        # Get client/config pairs from parent class
        client_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        
        # Add round number to each client's configuration
        for _, eval_ins in client_instructions:
            if eval_ins.config is None:
                eval_ins.config = {}
            eval_ins.config["round"] = server_round
        
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Report failures using the base class method
        self.report_failures(server_round, failures, "fit")
        
        # Call parent's aggregate_fit to perform Krum aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
        # Print Krum-specific information
        if "krum_selected_clients" in metrics:
            num_selected = metrics["krum_selected_clients"]
            method_name = "Multi-Krum" if self.multi_krum else "Krum"
            print(f"[Server] Round {server_round} -> {method_name} selected {num_selected} client{'s' if num_selected == 1 else 's'}")
        
        # Return the original parameters and metrics
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Report failures using the base class method
        self.report_failures(server_round, failures, "evaluate")
        
        # Call parent's aggregate_evaluate to perform the aggregation
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "evaluate")
        
        # Return the original result to preserve Flower's internal state
        return aggregated_loss, metrics
