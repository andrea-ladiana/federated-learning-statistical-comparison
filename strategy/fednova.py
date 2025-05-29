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



# FedNova Strategy Implementation
class FedNova(FedAvg):
    """Federated Normalized Averaging strategy implementation.
    
    Basato su: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
    https://proceedings.neurips.cc/paper/2020/file/564127c03caab942e503ee6f810f54fd-Paper.pdf
    
    FedNova normalizza i contributi di ogni client in base alle loro informazioni locali
    per affrontare il problema dell'inconsistenza degli aggiornamenti locali.
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

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training with specific FedNova instructions."""
        # Get client/config pairs from parent class
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Add specific FedNova config to track local steps
        for _, fit_ins in client_instructions:
            if fit_ins.config is None:
                fit_ins.config = {}
            # Notify clients to track local steps for FedNova
            fit_ins.config["track_local_steps"] = True
            fit_ins.config["round"] = server_round
        
        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using FedNova."""
        if not results:
            return None, {}

        # Extract weights, example counts, and local steps from clients
        weights_results = []
        for client_proxy, fit_res in results:
            # Extract parameters and number of examples
            weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            
            # Extract number of local steps from metrics
            # Default to 1 if not provided, but log a warning
            if fit_res.metrics and "num_local_steps" in fit_res.metrics:
                num_steps = fit_res.metrics["num_local_steps"]
            else:
                num_steps = 1
                print(f"[Server] Warning: Client {client_proxy.cid} did not report local steps. "
                      f"Defaulting to 1 for FedNova normalization.")
            
            weights_results.append((weights, num_examples, num_steps))

        # Get total number of examples
        total_examples = sum([num_examples for _, num_examples, _ in weights_results])
        
        if total_examples == 0:
            return None, {}
        
        # Calculate normalization factors for each client according to FedNova paper
        print(f"[Server] Round {server_round} → Calculating FedNova normalization factors:")
        
        norm_factors = []
        for i, (_, num_examples, num_steps) in enumerate(weights_results):
            # Calculate normalized update factor: (n_k/n) * (τ_k/n_k) = τ_k/n
            if num_examples > 0:
                # Convert num_steps to float before division (may be unnecessary, but safer)
                local_epochs_ratio = float(num_steps) / num_examples
                # Calculate p_k according to FedNova paper
                p_k = (num_examples / total_examples) * local_epochs_ratio
                norm_factors.append(p_k)
                print(f"[Server]   Client {i}: examples={num_examples}, steps={num_steps}, "
                      f"ratio={local_epochs_ratio:.6f}, factor={p_k:.6f}")
            else:
                norm_factors.append(0.0)
                print(f"[Server]   Client {i}: No examples, factor=0.0")
        
        # Normalize the factors to sum to 1
        total_norm = sum(norm_factors)
        if total_norm > 0:
            norm_factors = [factor / total_norm for factor in norm_factors]
            print(f"[Server] Normalized factors (sum={sum(norm_factors):.6f}):")
            for i, factor in enumerate(norm_factors):
                print(f"[Server]   Client {i}: norm_factor={factor:.6f}")
        else:
            # If all factors are zero, use uniform weights
            norm_factors = [1.0 / len(weights_results)] * len(weights_results)
            print(f"[Server] Using uniform factors due to zero normalization:")
            for i, factor in enumerate(norm_factors):
                print(f"[Server]   Client {i}: uniform_factor={factor:.6f}")
        
        # Perform weighted aggregation with normalized factors
        parameters_aggregated = ndarrays_to_parameters(aggregate_ndarrays_weighted(
            [weights for weights, _, _ in weights_results],
            norm_factors
        ))
        
        # Prepare metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(num_examples, metrics) for _, fit_res in results
                           for num_examples, metrics in [(fit_res.num_examples, fit_res.metrics)]]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            # Add FedNova-specific metrics
            metrics_aggregated["fednova_normalization_factors"] = str(norm_factors)
        
        return parameters_aggregated, metrics_aggregated


# Custom FedNova Strategy with metrics printing
class CustomFedNova(FedNova, BaseStrategy):
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
        
        # Call parent's aggregate_fit to perform the aggregation
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



