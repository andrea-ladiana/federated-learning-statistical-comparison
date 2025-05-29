from flwr.server.strategy import FedAvg
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
    """Federated Adam optimization strategy implementation.
    
    Based on: "Adaptive Federated Optimization" (Reddi et al., 2021)
    https://arxiv.org/abs/2003.00295
    
    Implements server-side adaptive optimization using Adam optimizer
    as described in Algorithm 1 of the paper.
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
        # Adam-specific parameters (as per the paper)
        server_learning_rate: float = 1e-3,  # η_s in the paper
        beta_1: float = 0.9,                 # β_1 in the paper
        beta_2: float = 0.999,               # β_2 in the paper  
        epsilon: float = 1e-4,               # ε in the paper (τ^2 = ε)
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
        self.server_learning_rate = server_learning_rate  # η_s
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Initialize server state
        self.server_parameters = None
        if initial_parameters is not None:
            self.server_parameters = parameters_to_ndarrays(initial_parameters)
            
        # Initialize Adam state variables (as per Algorithm 1 in the paper)
        self.m_t = None  # First moment vector
        self.v_t = None  # Second moment vector
        self.t = 0       # Timestep counter
        
        # Initialize Adam accumulators when we have initial parameters
        if self.server_parameters is not None:
            self.m_t = [np.zeros_like(param) for param in self.server_parameters]
            self.v_t = [np.zeros_like(param) for param in self.server_parameters]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using FedAdam as per Algorithm 1 in the paper."""
        if not results:
            return None, {}

        # Step 1: Compute weighted average of client updates (Δ_t in the paper)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        total_examples = sum([num_examples for _, num_examples in weights_results])
        if total_examples == 0:
            return None, {}
        
        # Compute weighted average of client model updates
        weighted_deltas = [
            np.zeros_like(weights_results[0][0][i]) for i in range(len(weights_results[0][0]))
        ]
        
        for weights, num_examples in weights_results:
            weight_factor = num_examples / total_examples
            for i, layer in enumerate(weights):
                weighted_deltas[i] += layer * weight_factor
        
        # Initialize server parameters if this is the first round
        if self.server_parameters is None and weighted_deltas:
            self.server_parameters = [np.copy(param) for param in weighted_deltas]
            self.m_t = [np.zeros_like(param) for param in self.server_parameters]
            self.v_t = [np.zeros_like(param) for param in self.server_parameters]
            # Return the initial aggregated parameters
            return ndarrays_to_parameters(self.server_parameters), {}
          # Ensure we have server parameters and Adam state is initialized
        if self.server_parameters is None:
            return None, {}
            
        if self.m_t is None or self.v_t is None:
            self.m_t = [np.zeros_like(param) for param in self.server_parameters]
            self.v_t = [np.zeros_like(param) for param in self.server_parameters]
        
        # Step 2: Compute "pseudo-gradient" (difference between aggregated update and current server model)
        # This is the key insight from the FedAdam paper: Δ_t = average of client updates
        # The pseudo-gradient is: g_t = Δ_t - x_t (where x_t is current server parameters)
        pseudo_gradient = []
        for i, (aggregated_update, server_param) in enumerate(zip(weighted_deltas, self.server_parameters)):
            # Following Algorithm 1: g_t = Δ_t - x_t
            grad = aggregated_update - server_param
            pseudo_gradient.append(grad)
        
        # Step 3: Increment timestep counter
        self.t += 1
        
        # Step 4: Update biased first moment estimate (m_t = β_1 * m_{t-1} + (1 - β_1) * g_t)
        for i, grad in enumerate(pseudo_gradient):
            self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * grad
        
        # Step 5: Update biased second moment estimate (v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2)
        for i, grad in enumerate(pseudo_gradient):
            self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * np.square(grad)
        
        # Step 6: Bias correction (as per Adam algorithm)
        m_hat = [m / (1 - self.beta_1 ** self.t) for m in self.m_t]
        v_hat = [v / (1 - self.beta_2 ** self.t) for v in self.v_t]
        
        # Step 7: Update server parameters (x_{t+1} = x_t - η_s * m_hat / (√v_hat + ε))
        for i in range(len(self.server_parameters)):
            self.server_parameters[i] = (
                self.server_parameters[i] - 
                self.server_learning_rate * m_hat[i] / (np.sqrt(v_hat[i]) + self.epsilon)
            )
        
        # Convert updated parameters to Flower format
        parameters_aggregated = ndarrays_to_parameters(self.server_parameters)
        
        # Prepare metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) 
                          for _, fit_res in results if fit_res.metrics is not None]
            if fit_metrics:
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        # Add FedAdam-specific metrics
        if len(pseudo_gradient) > 0:
            # Compute gradient norm for monitoring
            grad_norm = float(np.sqrt(sum(np.sum(g**2) for g in pseudo_gradient)))
            metrics_aggregated["fedadam_grad_norm"] = grad_norm
            metrics_aggregated["fedadam_server_lr"] = self.server_learning_rate
        
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
        
        # Call parent's aggregate_fit to perform FedAdam aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
        # Print FedAdam-specific information
        if "fedadam_grad_norm" in metrics:
            grad_norm = metrics["fedadam_grad_norm"]
            server_lr = metrics.get("fedadam_server_lr", self.server_learning_rate)
            print(f"[Server] Round {server_round} → FedAdam gradient norm: {grad_norm:.6f}, "
                  f"server LR: {server_lr}")
        
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
