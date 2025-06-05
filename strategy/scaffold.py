from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
import numpy as np
from utilities.utils import BaseStrategy, parameters_to_ndarrays, ndarrays_to_parameters, aggregate_ndarrays_weighted, logger


# Correct type for evaluate_fn
EvaluateFnType = Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]





# SCAFFOLD Strategy Implementation
class SCAFFOLD(FedAvg):
    """Stochastic Controlled Averaging for Federated Learning strategy implementation."""
    
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
        # Initialize server control variate
        self.server_control_variate = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training with server control variate."""
        # Get client/config pairs from parent class
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Add server control variate and round to each client's configuration
        for client_proxy, fit_ins in client_instructions:
            if fit_ins.config is None:
                fit_ins.config = {}
            
            current_config = fit_ins.config

            # Include server control variate in config if it exists
            if self.server_control_variate is not None:
                import pickle
                # self.server_control_variate is List[np.ndarray]
                # Convert to Parameters object to get List[bytes] via .tensors
                server_cv_params = ndarrays_to_parameters(self.server_control_variate)
                
                # Add metadata about the control variate for better logging
                cv_shapes = []
                for i, cv in enumerate(self.server_control_variate):
                    if hasattr(cv, 'shape'):
                        cv_shapes.append(f"layer{i}:{cv.shape}")
                    else:
                        cv_shapes.append(f"layer{i}:unknown_shape")
                
                # Add human-readable metadata
                current_config["scaffold_cv_info"] = f"Control variate with {len(self.server_control_variate)} layers: {', '.join(cv_shapes)}"
                
                # Still send the actual binary data (needed for the algorithm)
                pickled_server_cv_tensors = pickle.dumps(server_cv_params.tensors)
                current_config["server_control_variate"] = pickled_server_cv_tensors
            
            current_config["round"] = server_round
            
            # Log a clean version of the config without binary data
            clean_config = {}
            for key, value in current_config.items():
                if key == "server_control_variate" and isinstance(value, bytes):
                    clean_config[key] = f"<binary data of length {len(value)}>"
                else:
                    clean_config[key] = value
            
            logger.info(
                f"SCAFFOLD: Configuring client {client_proxy.cid} for round {server_round}. "
                f"Config: {clean_config}"
            )
        
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
        """Aggregate model updates using SCAFFOLD."""
        if not results:
            return None, {}

        # Extract weights and client information for parameter aggregation
        # Note: Using fit_res.parameters for model updates
        weights_results_params = []
        for client_proxy, fit_res in results:  # Use client_proxy instead of _ for better variable naming
            try:
                client_params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                if not client_params_ndarrays: # Check for empty parameters
                    logger.warning(f"SCAFFOLD: Client {client_proxy.cid} sent empty parameters. Skipping for model aggregation.")
                    continue
                weights_results_params.append((client_params_ndarrays, fit_res.num_examples))
            except Exception as e:
                logger.error(f"SCAFFOLD: Error processing parameters from client {client_proxy.cid}: {e}. Skipping.")
                continue
        
        if not weights_results_params:
            logger.warning("SCAFFOLD: No valid client parameters to aggregate. Skipping round.")
            return None, {}

        total_examples_params = sum([num_examples for _, num_examples in weights_results_params])
        if total_examples_params == 0:
            logger.warning("SCAFFOLD: Total examples for parameter aggregation is zero. Using FedAvg default.")
             # Fallback to super().aggregate_fit if no examples, which handles this.
            return super().aggregate_fit(server_round, results, failures)
            
        client_weights_params = [num_examples / total_examples_params for _, num_examples in weights_results_params]
        
        # Perform weighted aggregation of model parameters
        aggregated_model_ndarrays = aggregate_ndarrays_weighted(
            [weights for weights, _ in weights_results_params],
            client_weights_params
        )
        if not aggregated_model_ndarrays:
             logger.warning("SCAFFOLD: Model parameter aggregation resulted in empty ndarrays. Skipping CV update.")
             # Attempt to return None, {} or handle as per FedAvg's behavior for failed aggregation
             return (None, {}) if not self.accept_failures else (self.initial_parameters, {})

        parameters_aggregated = ndarrays_to_parameters(aggregated_model_ndarrays)
            
        # Initialize server control variate (c) if it's None
        if self.server_control_variate is None:
            if aggregated_model_ndarrays: # Ensure model_ndarrays is not empty
                self.server_control_variate = [
                    np.zeros_like(layer) for layer in aggregated_model_ndarrays
                ]
                logger.info(f"SCAFFOLD: Initialized server_control_variate with zeros in round {server_round}.")
            else:
                logger.error("SCAFFOLD: Failed to initialize server_control_variate, aggregated_model_ndarrays is empty.")
                # Return aggregated model parameters (if any) and empty metrics for CV part
                return parameters_aggregated, {} 

        # Update server control variate based on client control variate deltas
        if self.server_control_variate is not None: # Proceed only if server_control_variate is initialized
            for i, (client_proxy, fit_res) in enumerate(results):
                # Find corresponding weight for this client (results might be filtered vs weights_results_params)
                # This assumes 'results' and 'weights_results_params' (after filtering) correspond if client participated in param aggregation.
                # A safer way is to re-calculate weight or ensure results maps directly to client_weights_params
                # For simplicity, we use the index 'i' assuming 'results' maps to 'client_weights_params'
                # This might need adjustment if 'results' can be a subset of clients that provided 'weights_results_params'
                # due to earlier filtering. However, FedAvg usually passes all successful 'results'.
                current_client_weight = 0
                original_index_in_weights_results = -1
                for k_idx, (params_nd, num_ex) in enumerate(weights_results_params):
                    # This is a bit heuristic; ideally, map client_proxy.cid
                    # For now, assume order is preserved or use num_examples from fit_res
                    if fit_res.num_examples == num_ex and np.array_equal(parameters_to_ndarrays(fit_res.parameters)[0], params_nd[0]): # Basic check
                         current_client_weight = client_weights_params[k_idx]
                         original_index_in_weights_results = k_idx
                         break
                if original_index_in_weights_results == -1 and total_examples_params > 0 : # Fallback if direct match failed
                    current_client_weight = fit_res.num_examples / total_examples_params


                if fit_res.metrics and "control_variate_delta" in fit_res.metrics:
                    cv_delta_metric_value = fit_res.metrics["control_variate_delta"]
                    cv_delta_ndarrays = [] 
                    
                    if isinstance(cv_delta_metric_value, list) and all(isinstance(t, bytes) for t in cv_delta_metric_value):
                        try:
                            temp_params_cv_delta = Parameters(tensors=cv_delta_metric_value, tensor_type="numpy.ndarray")
                            cv_delta_ndarrays = parameters_to_ndarrays(temp_params_cv_delta)
                        except Exception as e:
                            logger.error(f"SCAFFOLD: Error deserializing control_variate_delta from client {client_proxy.cid}. Error: {e}")
                    elif cv_delta_metric_value is not None:
                        value_representation = ""
                        if isinstance(cv_delta_metric_value, bytes):
                            value_representation = f"bytes object of length {len(cv_delta_metric_value)}"
                        else:
                            try:
                                value_representation = str(cv_delta_metric_value)[:100]
                            except Exception:
                                value_representation = "[unrepresentable value]"
                        
                        logger.warning(
                            f"SCAFFOLD: control_variate_delta from client {client_proxy.cid} "
                            f"is not List[bytes]. Type: {type(cv_delta_metric_value)}. Value: {value_representation}"
                        )

                    if cv_delta_ndarrays: 
                        if len(self.server_control_variate) == len(cv_delta_ndarrays):
                            for j, delta_layer in enumerate(cv_delta_ndarrays):
                                if self.server_control_variate[j].shape == delta_layer.shape:
                                    self.server_control_variate[j] += current_client_weight * delta_layer
                                else:
                                    logger.warning(
                                        f"SCAFFOLD: Shape mismatch for control_variate_delta layer {j} from client {client_proxy.cid}. "
                                        f"Server CV shape: {self.server_control_variate[j].shape}, Client delta shape: {delta_layer.shape}"
                                    )
                        else:
                            logger.warning(
                                f"SCAFFOLD: Layer count mismatch for control_variate_delta from client {client_proxy.cid}. "
                                f"Server CV layers: {len(self.server_control_variate)}, Client delta layers: {len(cv_delta_ndarrays)}"
                            )
        
        # Prepare metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics_list = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
            if fit_metrics_list:
                 metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)
        
        if self.server_control_variate is not None and metrics_aggregated is not None:
            try:
                # Ensure all elements in server_control_variate are numpy arrays before flatten
                if all(isinstance(cv, np.ndarray) for cv in self.server_control_variate):
                    cv_norm = float(np.linalg.norm(np.concatenate([cv.flatten() for cv in self.server_control_variate])))
                    metrics_aggregated["scaffold_cv_norm"] = cv_norm
                else:
                    logger.warning("SCAFFOLD: server_control_variate contains non-ndarray elements, cannot compute norm.")
            except Exception as e:
                logger.warning(f"SCAFFOLD: Could not compute norm of server_control_variate. Error: {e}")

        return parameters_aggregated, metrics_aggregated


# Custom SCAFFOLD Strategy with metrics printing
class CustomSCAFFOLD(SCAFFOLD, BaseStrategy):
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

