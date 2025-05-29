from flwr.server.strategy.fedavgm import FedAvgM
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utils import BaseStrategy

# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]

# Custom FedAvgM Strategy with metrics printing
class CustomFedAvgM(FedAvgM, BaseStrategy):
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
        server_momentum: float = 0.9,
    ):
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
            server_momentum=server_momentum,
        )
        
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
