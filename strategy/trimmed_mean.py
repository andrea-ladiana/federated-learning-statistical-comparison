from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utilities.utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters


# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]



# TrimmedMean Strategy Implementation
class TrimmedMean(FedAvg):
    """Trimmed Mean aggregation strategy implementation.
    
    Based on the paper: https://proceedings.mlr.press/v80/yin18a/yin18a.pdf
    A Byzantine-robust strategy that performs coordinate-wise trimming of extreme values.
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
        # TrimmedMean-specific parameters
        beta: float = 0.1,  # Fraction of values to trim from each end
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
        
        # TrimmedMean-specific attributes
        self.beta = beta
    
    def _compute_trimmed_mean(self, values: np.ndarray) -> np.ndarray:
        """Compute trimmed mean by removing extreme values.
        
        Args:
            values: Array of shape (num_clients, ...) where the first dimension
                   represents different clients' values for the same parameter/layer.
        
        Returns:
            Trimmed mean with the same shape as a single client's values.
        """
        # Get number of clients
        num_clients = values.shape[0]
        
        # Calculate how many values to trim from each end
        k = int(self.beta * num_clients)
        
        if k * 2 >= num_clients:
            # If we would trim too many values, fall back to median
            # This happens when beta is too large or there are too few clients
            return np.median(values, axis=0)
        
        # Sort values along client dimension (axis 0)
        sorted_values = np.sort(values, axis=0)
        
        # Select values after trimming
        trimmed_values = sorted_values[k:num_clients-k]
        
        # Compute mean of remaining values
        return np.mean(trimmed_values, axis=0)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Trimmed Mean."""
        if not results:
            return None, {}

        # Extract weights and client information
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Extract client models
        client_models = [weights for weights, _ in weights_results]
        client_examples = [num_examples for _, num_examples in weights_results]
        
        # Need at least 2 clients to perform trimming
        if len(client_models) < 2:
            print(f"[Server] Warning: Not enough clients for Trimmed Mean (have {len(client_models)}). Using default aggregation.")
            return super().aggregate_fit(server_round, results, failures)
        
        # Apply Trimmed Mean to each layer/parameter separately
        aggregated_ndarrays = []
        
        # For each layer/parameter in the model
        for layer_idx in range(len(client_models[0])):
            # Collect the same layer from all clients
            layer_values = np.array([model[layer_idx] for model in client_models])
            
            # Apply trimmed mean to this layer
            aggregated_layer = self._compute_trimmed_mean(layer_values)
            
            # Add to the list of aggregated layers
            aggregated_ndarrays.append(aggregated_layer)
        
        # Convert aggregated parameters to Flower format
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Prepare metrics with standard aggregation
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(num_examples, metrics) for _, fit_res in results
                           for num_examples, metrics in [(fit_res.num_examples, fit_res.metrics)]]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            # Add TrimmedMean-specific metrics
            metrics_aggregated["trimmed_fraction"] = self.beta
            metrics_aggregated["values_trimmed"] = int(self.beta * len(client_models) * 2)  # Both ends
        
        return parameters_aggregated, metrics_aggregated


# Custom TrimmedMean Strategy with metrics printing
class CustomTrimmedMean(TrimmedMean, BaseStrategy):
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
        
        # Call parent's aggregate_fit to perform TrimmedMean aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
        # Print TrimmedMean-specific information
        if "values_trimmed" in metrics:
            values_trimmed = metrics["values_trimmed"]
            print(f"[Server] Round {server_round} -> Trimmed Mean removed {values_trimmed} extreme values " +
                  f"({self.beta*100:.1f}% from each end)")
        
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
