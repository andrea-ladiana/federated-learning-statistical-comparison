from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters, aggregate_ndarrays_weighted


# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]



# FedAdam Strategy Implementation
class FedAdam(FedAvg):
    """Federated Adam optimization strategy implementation."""
    
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
        # Adam-specific parameters
        server_learning_rate: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        epsilon: float = 1e-8,
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
        
        # Adam optimizer parameters
        self.server_learning_rate = server_learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Adam state initialization
        self.current_parameters = None
        if initial_parameters is not None:
            self.current_parameters = parameters_to_ndarrays(initial_parameters)
            
        # Initialize momentum and variance accumulators
        self.m_t = None  # First moment vector (momentum)
        self.v_t = None  # Second moment vector (variance)
        self.t = 0  # Timestep counter
        
        if self.current_parameters is not None:
            self.m_t = [np.zeros_like(param) for param in self.current_parameters]
            self.v_t = [np.zeros_like(param) for param in self.current_parameters]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Adam optimization."""
        if not results:
            return None, {}

        # Extract weights and client information
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Calculate weighted average of model updates
        total_examples = sum([num_examples for _, num_examples in weights_results])
        
        # Create weighted average for basic aggregation
        aggregated_ndarrays = [
            np.zeros_like(weights_results[0][0][i]) for i in range(len(weights_results[0][0]))
        ]
        
        for weights, num_examples in weights_results:
            weight_factor = num_examples / total_examples
            for i, layer in enumerate(weights):
                aggregated_ndarrays[i] += layer * weight_factor
                
        # If this is the first round, initialize current parameters
        if self.current_parameters is None and aggregated_ndarrays:
            self.current_parameters = [np.copy(param) for param in aggregated_ndarrays]
            self.m_t = [np.zeros_like(param) for param in self.current_parameters]
            self.v_t = [np.zeros_like(param) for param in self.current_parameters]
        
        # Increment timestep counter
        self.t += 1
        
        # Apply Adam update
        if self.current_parameters is not None and aggregated_ndarrays:
            # Calculate pseudogradient (difference between current and new parameters)
            pseudo_gradient = []
            for i, (current, aggregated) in enumerate(zip(self.current_parameters, aggregated_ndarrays)):
                # The "gradient" is the difference between current parameters and aggregated parameters
                pseudo_gradient.append(current - aggregated)
            
            # Initialize momentum and variance if they don't exist yet
            if self.m_t is None or self.v_t is None:
                self.m_t = [np.zeros_like(param) for param in self.current_parameters]
                self.v_t = [np.zeros_like(param) for param in self.current_parameters]
            
            # Update momentum (first moment)
            for i, grad in enumerate(pseudo_gradient):
                self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * grad
            
            # Update variance (second moment)
            for i, grad in enumerate(pseudo_gradient):
                self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * np.square(grad)
            
            # Bias correction
            m_t_hat = [m / (1 - self.beta_1 ** self.t) for m in self.m_t]
            v_t_hat = [v / (1 - self.beta_2 ** self.t) for v in self.v_t]
            
            # Update parameters
            for i in range(len(self.current_parameters)):
                self.current_parameters[i] -= self.server_learning_rate * m_t_hat[i] / (np.sqrt(v_t_hat[i]) + self.epsilon)
        
        # Convert updated parameters to Flower format
        if self.current_parameters is None:
            # Return empty parameters if current_parameters is None
            parameters_aggregated = Parameters(tensors=[], tensor_type="numpy.ndarray")
        else:
            parameters_aggregated = ndarrays_to_parameters(self.current_parameters)
        
        # Prepare metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) 
                          for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        return parameters_aggregated, metrics_aggregated


# Custom FedAdam Strategy with metrics printing
class CustomFedAdam(FedAdam, BaseStrategy):
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
        
        # Call parent's aggregate_fit to perform the aggregation with Adam
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
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