from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utilities.utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters


# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]



# Bulyan Strategy Implementation
class Bulyan(FedAvg):
    """Bulyan aggregation strategy implementation.
    
    Based on: https://pmc.ncbi.nlm.nih.gov/articles/PMC9141408/pdf/entropy-24-00686.pdf
    Combines Multi-Krum with coordinate-wise trimmed mean for enhanced Byzantine resistance.
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
        # Bulyan-specific parameters
        num_byzantine: int = 0,          # Expected number of Byzantine clients
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
        
        # Bulyan-specific attributes
        self.num_byzantine = num_byzantine
    
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
    
    def _select_models_with_krum(
        self, 
        client_models: List[List[np.ndarray]], 
        client_examples: List[int]
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Use Multi-Krum to select candidate models."""
        num_clients = len(client_models)
        f = self.num_byzantine
        
        # Need at least 4f+3 clients for Bulyan to be well-defined
        min_clients_required = 4 * f + 3
        if num_clients < min_clients_required:
            print(f"[Server] Warning: Not enough clients for Bulyan (need {min_clients_required}, have {num_clients}). Using Multi-Krum instead.")
            # If we don't have enough clients for Bulyan, use Multi-Krum
            # but ensure we select at least one client
            min_clients_required = 2 * f + 3
            if num_clients < min_clients_required:
                print(f"[Server] Warning: Not enough clients for Multi-Krum (need {min_clients_required}, have {num_clients}). Using default aggregation.")
                return [], []  # Return empty lists instead of None
        
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
        
        # For Bulyan, we need to select n-2f models using Multi-Krum
        m = n - 2 * f
        m = min(m, n)  # Ensure m doesn't exceed number of clients
        
        # Get indices of clients with lowest scores
        selected_indices = np.argsort(scores)[:m]
        
        # Get selected models and their weights based on example counts
        selected_models = [client_models[i] for i in selected_indices]
        selected_examples = [client_examples[i] for i in selected_indices]
        
        return selected_models, selected_examples
    
    def _apply_bulyan_trimmed_mean(self, selected_models: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Apply coordinate-wise trimmed mean on selected models."""
        # Number of selected models after Multi-Krum
        num_selected = len(selected_models)
        f = self.num_byzantine
        
        # Bulyan requires trimming 2f values from each coordinate
        # (f largest and f smallest)
        trim_count = 2 * f
        
        # If we would trim too many values, fall back to median
        if trim_count >= num_selected:
            print(f"[Server] Warning: Bulyan trim count ({trim_count}) is too large for {num_selected} selected models. Using median.")
            # Create a list to store aggregated parameters
            aggregated_ndarrays = []
            
            # Apply median to each parameter/layer
            for layer_idx in range(len(selected_models[0])):
                # Collect same layer from all selected models
                layer_values = np.array([model[layer_idx] for model in selected_models])
                # Apply median
                aggregated_layer = np.median(layer_values, axis=0)
                # Add to list
                aggregated_ndarrays.append(aggregated_layer)
                
            return aggregated_ndarrays
        
        # Create a list to store aggregated parameters
        aggregated_ndarrays = []
        
        # Apply trimmed mean to each parameter/layer
        for layer_idx in range(len(selected_models[0])):
            # Get the layer shape from the first model
            layer_shape = selected_models[0][layer_idx].shape
            
            # Create a flattened version of this layer from all models
            flattened_layers = [model[layer_idx].flatten() for model in selected_models]
            
            # Reshape to (num_selected, num_parameters)
            stacked_layers = np.vstack(flattened_layers)
            
            # Apply trimmed mean to each parameter/coordinate
            aggregated_flat = np.zeros(stacked_layers.shape[1])
            
            # For each parameter/coordinate
            for param_idx in range(stacked_layers.shape[1]):
                # Extract values for this parameter across all models
                param_values = stacked_layers[:, param_idx]
                
                # Sort values
                sorted_values = np.sort(param_values)
                
                # Apply trimming (remove f smallest and f largest)
                trimmed_values = sorted_values[f:num_selected-f]
                
                # Calculate mean of remaining values
                aggregated_flat[param_idx] = np.mean(trimmed_values)
            
            # Reshape back to original layer shape
            aggregated_layer = aggregated_flat.reshape(layer_shape)
            
            # Add to list of aggregated layers
            aggregated_ndarrays.append(aggregated_layer)
        
        return aggregated_ndarrays
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Bulyan."""
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
        
        # Step 1: Apply Multi-Krum to select candidate models
        selected_models, selected_examples = self._select_models_with_krum(client_models, client_examples)
        
        # If Krum selection failed (empty lists), fall back to standard FedAvg
        if len(selected_models) == 0:
            return super().aggregate_fit(server_round, results, failures)
        
        # Step 2: Apply Bulyan trimmed mean on selected models
        aggregated_ndarrays = self._apply_bulyan_trimmed_mean(selected_models)
        
        # Convert aggregated parameters to Flower format
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Prepare metrics with standard aggregation
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(num_examples, metrics) for _, fit_res in results
                           for num_examples, metrics in [(fit_res.num_examples, fit_res.metrics)]]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            # Add Bulyan-specific metrics
            metrics_aggregated["bulyan_selected_models"] = len(selected_models)
            metrics_aggregated["bulyan_trimmed_params"] = 2 * self.num_byzantine  # f smallest + f largest
        
        return parameters_aggregated, metrics_aggregated


# Custom Bulyan Strategy with metrics printing
class CustomBulyan(Bulyan, BaseStrategy):
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
        
        # Call parent's aggregate_fit to perform Bulyan aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
        # Print Bulyan-specific information
        if "bulyan_selected_models" in metrics and "bulyan_trimmed_params" in metrics:
            models_selected = metrics["bulyan_selected_models"]
            params_trimmed = metrics["bulyan_trimmed_params"]
            print(f"[Server] Round {server_round} -> Bulyan selected {models_selected} models " +
                  f"and trimmed {params_trimmed} values per parameter " +
                  f"({self.num_byzantine} smallest, {self.num_byzantine} largest)")
        
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