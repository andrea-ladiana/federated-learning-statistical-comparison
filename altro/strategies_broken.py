from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from strategy.bulyan import CustomBulyan
from strategy.trimmed_mean import CustomTrimmedMean
from strategy.krum import CustomKrum
from strategy.fedadam import CustomFedAdam
from strategy.fedatt import CustomFedAtt
from strategy.scaffold import CustomSCAFFOLD
from strategy.fedavg import CustomFedAvg
from strategy.fedprox import CustomFedProx
from strategy.fednova import CustomFedNova
from strategy.fedavgm import CustomFedAvgM  # Import della nuova strategia FedAvgM
from strategy.baseline_wrappers import (
    BaseDashaWrapper, BaseDepthFLWrapper, BaseHeteroFLWrapper,
    BaseFedMetaWrapper, BaseFedPerWrapper, BaseFjordWrapper,
    BaseFlandersWrapper, BaseFedOptWrapper
)
import numpy as np

# Import from utils instead of defining here
from utils import (
    BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters,
    safe_aggregate_parameters, aggregate_ndarrays_weighted,
    weighted_average, EvaluateFnType, logger
)

# Factory function to create different strategy instances
def create_strategy(
    strategy_name: str,
    initial_parameters: Parameters,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 1,
    min_available_clients: int = 1,
    fraction_evaluate: float = 0.5,
    min_evaluate_clients: int = 1,
    evaluate_fn: Optional[EvaluateFnType] = None,
    **kwargs
) -> FedAvg:
    """
    Factory function to create different strategy instances.
    
    Args:
        strategy_name: The name of the strategy to create
        initial_parameters: Initial model parameters
        evaluate_fn: Function that accepts (round, parameters, config) and returns evaluation metrics
        Other standard Flower strategy parameters
        
    Returns:
        An instance of the requested strategy
    """
    # Common parameters for all strategies
    common_kwargs = {
        "initial_parameters": initial_parameters,
        "fraction_fit": fraction_fit,
        "min_fit_clients": min_fit_clients,
        "min_available_clients": min_available_clients,
        "fraction_evaluate": fraction_evaluate,
        "min_evaluate_clients": min_evaluate_clients,
        "evaluate_fn": evaluate_fn,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
        **kwargs
    }
    
    if strategy_name == "fedavg":
        return CustomFedAvg(**common_kwargs)
    elif strategy_name == "fedavgm":
        # Parametro specifico per FedAvgM
        server_momentum = kwargs.get("server_momentum", 0.9)
        return CustomFedAvgM(**common_kwargs, server_momentum=server_momentum)
    elif strategy_name == "fedprox":
        # Parametro specifico per FedProx
        proximal_mu = kwargs.get("proximal_mu", 0.01)
        return CustomFedProx(**common_kwargs, proximal_mu=proximal_mu)
    elif strategy_name == "fednova":
        return CustomFedNova(**common_kwargs)
    elif strategy_name == "scaffold":
        return CustomSCAFFOLD(**common_kwargs)
    elif strategy_name == "fedadam":
        # Parametro specifico per FedAdam
        server_learning_rate = kwargs.get("server_learning_rate", 0.1)
        return CustomFedAdam(**common_kwargs, server_learning_rate=server_learning_rate)
    elif strategy_name == "fedatt":
        return CustomFedAtt(**common_kwargs)    elif strategy_name == "krum":
        # Parametro specifico per Krum
        num_byzantine = kwargs.get("num_byzantine", 0)
        return CustomKrum(**common_kwargs, num_byzantine=num_byzantine)
    elif strategy_name == "trimmedmean":
        # Parametro specifico per TrimmedMean
        beta = kwargs.get("beta", 0.1)
        return CustomTrimmedMean(**common_kwargs, beta=beta)
    elif strategy_name == "bulyan":
        # Parametro specifico per Bulyan
        num_byzantine = kwargs.get("num_byzantine", 0)
        return CustomBulyan(**common_kwargs, num_byzantine=num_byzantine)
    # Baseline wrapper strategies
    elif strategy_name == "dasha":
        # DASHA specific parameters
        step_size = kwargs.get("step_size", 0.5)
        compressor_coords = kwargs.get("compressor_coords", 10)
        return BaseDashaWrapper(**common_kwargs, step_size=step_size, compressor_coords=compressor_coords)
    elif strategy_name == "depthfl":
        # DepthFL specific parameters
        alpha = kwargs.get("alpha", 0.75)
        tau = kwargs.get("tau", 0.6)
        return BaseDepthFLWrapper(**common_kwargs, alpha=alpha, tau=tau)
    elif strategy_name == "heterofl":
        # HeteroFL specific parameters
        return BaseHeteroFLWrapper(**common_kwargs)
    elif strategy_name == "fedmeta":
        # FedMeta specific parameters
        beta = kwargs.get("beta", 1.0)
        return BaseFedMetaWrapper(**common_kwargs, beta=beta)
    elif strategy_name == "fedper":
        # FedPer specific parameters
        return BaseFedPerWrapper(**common_kwargs)
    elif strategy_name == "fjord":
        # FjORD specific parameters
        return BaseFjordWrapper(**common_kwargs)
    elif strategy_name == "flanders":
        # FLANDERS specific parameters
        to_keep = kwargs.get("to_keep", 0.6)
        return BaseFlandersWrapper(**common_kwargs, to_keep=to_keep)
    elif strategy_name == "fedopt":
        # FedOpt specific parameters
        tau = kwargs.get("tau", 1e-3)
        beta_1 = kwargs.get("beta_1", 0.9)
        beta_2 = kwargs.get("beta_2", 0.99)
        eta = kwargs.get("eta", 1e-3)
        eta_l = kwargs.get("eta_l", 1e-3)
        return BaseFedOptWrapper(**common_kwargs, tau=tau, beta_1=beta_1, beta_2=beta_2, eta=eta, eta_l=eta_l)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

# NOTE: To avoid circular imports, strategy implementation files should import 
# utility functions directly from utils.py instead of from strategies.py:
#
# Instead of: from ..strategies import BaseStrategy, parameters_to_ndarrays, ...
# Use: from ..utils import BaseStrategy, parameters_to_ndarrays, ...
#
# This prevents the import error: "ImportError: attempted relative import beyond top-level package"
