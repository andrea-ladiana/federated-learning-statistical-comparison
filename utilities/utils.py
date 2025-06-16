from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from flwr.common.typing import NDArrays
import abc
import numpy as np
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strategies")

# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]

def parameters_to_ndarrays(parameters: Parameters) -> List[np.ndarray]:
    """Convert parameters to numpy ndarrays with improved error handling."""
    try:
        # Assuming parameters.tensors contains bytes in .npy format
        return [np.load(io.BytesIO(param_bytes), allow_pickle=False) for param_bytes in parameters.tensors]
    except Exception as e:
        logger.error(f"Error converting parameters to ndarrays: {str(e)}")
        # Return empty list or default parameters if conversion fails
        return []

def ndarrays_to_parameters(ndarrays: List[np.ndarray]) -> Parameters:
    """Convert numpy ndarrays to parameters with improved error handling."""
    try:
        tensors = []
        for ndarray in ndarrays:
            # Ensure the array is contiguous and correct dtype
            ndarray_contiguous = np.ascontiguousarray(ndarray, dtype=np.float32)
            bytes_io = io.BytesIO()
            np.save(bytes_io, ndarray_contiguous, allow_pickle=False)
            tensors.append(bytes_io.getvalue())
        return Parameters(tensors=tensors, tensor_type="numpy.ndarray")
    except Exception as e:
        logger.error(f"Error converting ndarrays to parameters: {str(e)}")
        # Return empty parameters if conversion fails
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

# Additional helper for safer parameter handling
def safe_aggregate_parameters(parameter_list: List[Parameters]) -> Parameters:
    """Safely aggregate multiple parameter objects into one."""
    if not parameter_list:
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

    try:
        # Convert all parameters to ndarrays
        all_ndarrays = [parameters_to_ndarrays(p) for p in parameter_list]
        # Filter out empty results
        valid_ndarrays = [arr for arr in all_ndarrays if len(arr) > 0]

        if not valid_ndarrays:
            return Parameters(tensors=[], tensor_type="numpy.ndarray")

        # Validate that all arrays have matching shapes
        ref_shapes = [layer.shape for layer in valid_ndarrays[0]]
        for ndarrays in valid_ndarrays[1:]:
            if len(ndarrays) != len(ref_shapes):
                logger.error("Shape mismatch detected in safe_aggregate_parameters")
                return Parameters(tensors=[], tensor_type="numpy.ndarray")
            for layer, ref_shape in zip(ndarrays, ref_shapes):
                if layer.shape != ref_shape:
                    logger.error("Shape mismatch detected in safe_aggregate_parameters")
                    return Parameters(tensors=[], tensor_type="numpy.ndarray")

        # Use the first valid result as a template
        aggregated = [np.zeros_like(layer) for layer in valid_ndarrays[0]]

        # Simple averaging
        for ndarrays in valid_ndarrays:
            for i, layer in enumerate(ndarrays):
                aggregated[i] += layer / len(valid_ndarrays)

        return ndarrays_to_parameters(aggregated)
    except Exception as e:
        logger.error(f"Error in safe_aggregate_parameters: {str(e)}")
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

def aggregate_ndarrays_weighted(weights: List[List[np.ndarray]], normalization_factors: List[float]) -> List[np.ndarray]:
    """Aggregate model weights using weighted averaging with normalization factors and error handling."""
    try:
        if not weights or not normalization_factors or len(weights) != len(normalization_factors):
            logger.error("Invalid inputs to aggregate_ndarrays_weighted")
            return []

        # Ensure all normalization factors are non-negative
        if not all(f >= 0 for f in normalization_factors):
            logger.error("Normalization factors must be non-negative")
            return []

        # Normalize factors so that they sum to 1
        total_factor = sum(normalization_factors)
        if total_factor <= 0:
            logger.error("Sum of normalization factors must be positive")
            return []
        normalization_factors = [f / total_factor for f in normalization_factors]
        
        factors = np.asarray(normalization_factors, dtype=np.float32)

        aggregated_ndarrays = []
        try:
            layers = list(zip(*weights))
            for layer_idx, layer_weights in enumerate(layers):
                try:
                    stack = np.stack([
                        np.asarray(w, dtype=np.float32) for w in layer_weights
                    ])
                    aggregated = np.average(stack, axis=0, weights=factors)
                    aggregated_ndarrays.append(aggregated)
                except Exception as e:
                    logger.warning(
                        f"Error processing layer {layer_idx}: {str(e)}"
                    )
                    aggregated_ndarrays.append(np.zeros_like(layer_weights[0]))
        except Exception as e:
            logger.error(f"Error stacking weights: {str(e)}")
            return []

        return aggregated_ndarrays
    except Exception as e:
        logger.error(f"Error in aggregate_ndarrays_weighted: {str(e)}")
        return []

# Function to calculate weighted average of metrics
def weighted_average(metrics: List[Tuple[int, Optional[Dict[str, Scalar]]]]) -> Dict[str, Scalar]:
    """Compute weighted average for an arbitrary set of metrics.

    Only tuples with a positive ``num_examples`` and a non ``None`` metrics
    dictionary contribute to the aggregation.
    """
    if not metrics:
        logger.warning("No metrics provided to weighted_average")
        return {}

    # Filter out invalid entries
    valid_metrics = [
        (num_examples, m)
        for num_examples, m in metrics
        if m is not None and num_examples > 0
    ]

    if not valid_metrics:
        logger.warning("No valid metrics to aggregate in weighted_average")
        return {}

    total_examples = sum(num_examples for num_examples, _ in valid_metrics)

    if total_examples == 0:
        logger.warning("Total number of examples is zero in weighted_average")
        return {}

    aggregated: Dict[str, Scalar] = {}
    all_keys = set().union(*(m.keys() for _, m in valid_metrics))
    for key in all_keys:
        values = [float(m.get(key, 0.0)) * num_examples for num_examples, m in valid_metrics]
        aggregated[key] = float(sum(values)) / total_examples

    return aggregated

# Base strategy class with common functionality
class BaseStrategy(abc.ABC):
    """Base class for all FL strategies with common reporting and metrics handling."""
    
    def report_failures(self, server_round: int, failures: List[Union[Tuple[Any, Any], BaseException]], phase: str):
        """Report client failures during a round."""
        if failures:
            print(f"[Server] Round {server_round}: {len(failures)} clients failed during {phase}")
            for failure in failures:
                if isinstance(failure, BaseException):
                    print(f"[Server] Client failure: {str(failure)}")
                else:
                    client, _ = failure
                    print(f"[Server] Client {client.cid} failed")
    def report_metrics(self, server_round: int, metrics: Dict[str, Scalar], phase: str):
        """Report aggregated metrics."""
        if metrics:
            ordered_keys = ["accuracy", "loss", "precision", "recall", "f1"]
            metric_parts = []
            for key in ordered_keys:
                if key in metrics:
                    metric_parts.append(f"{key}={float(metrics[key]):.4f}")
            remaining = [k for k in metrics.keys() if k not in ordered_keys]
            for key in remaining:
                metric_parts.append(f"{key}={float(metrics[key]):.4f}")
            metrics_str = ", ".join(metric_parts)
            print(f"[Server] Round {server_round} {phase} -> {metrics_str}")
