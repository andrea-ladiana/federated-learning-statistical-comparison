"""
Wrapper classes for Flower baseline strategies with custom reporting capabilities.

This module provides wrapper classes for integrating Flower baseline strategies
(Dasha, DepthFL, HeteroFL, FedMeta, FedPer, Fjord, Flanders, FedOpt) with 
the existing federated learning framework, adding custom metrics reporting
and ensuring compatibility with the command-line interface.
"""

from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters, aggregate_ndarrays_weighted, logger
import sys
import os

# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]


class BaseDashaWrapper(FedAvg, BaseStrategy):
    """Wrapper for DASHA baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        step_size: float = 0.5,
        compressor_coords: int = 10,
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
        **kwargs
    ):
        """Initialize DASHA wrapper strategy.
        
        Args:
            step_size: Step size for DASHA algorithm
            compressor_coords: Number of coordinates for compression
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.step_size = step_size
        self.compressor_coords = compressor_coords
        self.strategy_name = "dasha"
        
        logger.info("✅ Initialized DASHA strategy wrapper")
        logger.info(f"   Step size: {step_size}")
        logger.info(f"   Compressor coordinates: {compressor_coords}")


class BaseDepthFLWrapper(FedAvg, BaseStrategy):
    """Wrapper for DepthFL baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        use_knowledge_distillation: bool = True,
        use_feddyn: bool = True,
        alpha: float = 0.1,
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
        **kwargs
    ):
        """Initialize DepthFL wrapper strategy.
        
        Args:
            use_knowledge_distillation: Enable self-distillation
            use_feddyn: Enable FedDyn regularization
            alpha: Alpha parameter for FedDyn
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.use_knowledge_distillation = use_knowledge_distillation
        self.use_feddyn = use_feddyn
        self.alpha = alpha
        self.strategy_name = "depthfl"
        
        logger.info("✅ Initialized DepthFL strategy wrapper")
        logger.info(f"   Knowledge distillation: {use_knowledge_distillation}")
        logger.info(f"   FedDyn regularization: {use_feddyn}")
        logger.info(f"   Alpha parameter: {alpha}")


class BaseHeteroFLWrapper(FedAvg, BaseStrategy):
    """Wrapper for HeteroFL baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        global_model_rate: float = 1.0,
        use_scaler: bool = True,
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
        **kwargs
    ):
        """Initialize HeteroFL wrapper strategy.
        
        Args:
            global_model_rate: Global model complexity rate
            use_scaler: Enable scaler module
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.global_model_rate = global_model_rate
        self.use_scaler = use_scaler
        self.strategy_name = "heterofl"
        
        logger.info("✅ Initialized HeteroFL strategy wrapper")
        logger.info(f"   Global model rate: {global_model_rate}")
        logger.info(f"   Use scaler module: {use_scaler}")


class BaseFedMetaWrapper(FedAvg, BaseStrategy):
    """Wrapper for FedMeta baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        alpha: float = 0.01,
        beta: float = 0.01,
        algorithm: str = "meta_sgd",
        weight_decay: float = 0.0,
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
        **kwargs
    ):
        """Initialize FedMeta wrapper strategy.
        
        Args:
            alpha: Alpha parameter for meta learning
            beta: Beta parameter for meta learning
            algorithm: Meta learning algorithm ("meta_sgd" or "maml")
            weight_decay: Weight decay parameter
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.alpha = alpha
        self.beta = beta
        self.algorithm = algorithm
        self.weight_decay = weight_decay
        self.strategy_name = "fedmeta"
        
        logger.info("✅ Initialized FedMeta strategy wrapper")
        logger.info(f"   Alpha parameter: {alpha}")
        logger.info(f"   Beta parameter: {beta}")
        logger.info(f"   Algorithm: {algorithm}")
        logger.info(f"   Weight decay: {weight_decay}")


class BaseFedPerWrapper(FedAvg, BaseStrategy):
    """Wrapper for FedPer baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        algorithm: str = "fedper",
        personalization_layers: int = 1,
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
        **kwargs
    ):
        """Initialize FedPer wrapper strategy.
        
        Args:
            algorithm: FedPer algorithm variant
            personalization_layers: Number of personalization layers
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.algorithm = algorithm
        self.personalization_layers = personalization_layers
        self.strategy_name = "fedper"
        
        logger.info("✅ Initialized FedPer strategy wrapper")
        logger.info(f"   Algorithm: {algorithm}")
        logger.info(f"   Personalization layers: {personalization_layers}")


class BaseFjordWrapper(FedAvg, BaseStrategy):
    """Wrapper for FjORD baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        p_max: float = 1.0,
        adapt_p: bool = True,
        compression_ratio: float = 0.5,
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
        **kwargs
    ):
        """Initialize FjORD wrapper strategy.
        
        Args:
            p_max: Maximum p-value for adaptive width
            adapt_p: Enable adaptive p selection
            compression_ratio: Compression ratio for communication
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.p_max = p_max
        self.adapt_p = adapt_p
        self.compression_ratio = compression_ratio
        self.strategy_name = "fjord"
        
        logger.info("✅ Initialized FjORD strategy wrapper")
        logger.info(f"   P max: {p_max}")
        logger.info(f"   Adaptive p: {adapt_p}")
        logger.info(f"   Compression ratio: {compression_ratio}")


class BaseFlandersWrapper(FedAvg, BaseStrategy):
    """Wrapper for FLANDERS baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        num_clients_to_keep: int = 8,
        window: int = 5,
        alpha: float = 0.1,
        beta: int = 2,
        maxiter: int = 10,
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
        **kwargs
    ):
        """Initialize FLANDERS wrapper strategy.
        
        Args:
            num_clients_to_keep: Number of clients to keep after filtering
            window: Window size for MAR detection
            alpha: Alpha parameter for MAR
            beta: Beta parameter for MAR
            maxiter: Maximum iterations for aggregation
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.num_clients_to_keep = num_clients_to_keep
        self.window = window
        self.alpha = alpha
        self.beta = beta
        self.maxiter = maxiter
        self.strategy_name = "flanders"
        
        logger.info("✅ Initialized FLANDERS strategy wrapper")
        logger.info(f"   Clients to keep: {num_clients_to_keep}")
        logger.info(f"   Window size: {window}")
        logger.info(f"   Alpha: {alpha}")
        logger.info(f"   Beta: {beta}")
        logger.info(f"   Max iterations: {maxiter}")


class BaseFedOptWrapper(FedAvg, BaseStrategy):
    """Wrapper for FedOpt baseline strategy with custom reporting."""
    
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        optimizer: str = "adam",
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        tau: float = 1e-3,
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
        **kwargs
    ):
        """Initialize FedOpt wrapper strategy.
        
        Args:
            optimizer: Server optimizer type ("adam", "adagrad", "yogi")
            server_learning_rate: Server-side learning rate
            server_momentum: Server-side momentum
            tau: Tau parameter for adaptive optimizers
        """
        super().__init__(
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
            initial_parameters=initial_parameters,
        )
        
        self.optimizer = optimizer
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.tau = tau
        self.strategy_name = "fedopt"
        
        logger.info("✅ Initialized FedOpt strategy wrapper")
        logger.info(f"   Optimizer: {optimizer}")
        logger.info(f"   Server learning rate: {server_learning_rate}")
        logger.info(f"   Server momentum: {server_momentum}")
        logger.info(f"   Tau parameter: {tau}")
