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




# FedAtt Strategy Implementation
class FedAtt(FedAvg):
    """Federated Attentive Message Passing strategy implementation.
    
    Based on the paper: https://arxiv.org/pdf/1812.07108
    Uses an attention mechanism to assign weights to clients during aggregation.
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
        # FedAtt-specific parameters
        attention_temperature: float = 1.0,
        similarity_metric: str = "cosine",
        min_attention_value: float = 0.0,
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
        
        # FedAtt-specific attributes
        self.attention_temperature = attention_temperature
        self.similarity_metric = similarity_metric
        self.min_attention_value = min_attention_value
        
        # Store current global model for reference
        self.current_parameters = None
        if initial_parameters is not None:
            self.current_parameters = parameters_to_ndarrays(initial_parameters)
    
    def _calculate_similarity(self, model_a: List[np.ndarray], model_b: List[np.ndarray]) -> float:
        """Calculate the similarity between two models based on the chosen metric."""
        if self.similarity_metric == "cosine":
            # Flatten model parameters into single vectors
            vec_a = np.concatenate([param.flatten() for param in model_a])
            vec_b = np.concatenate([param.flatten() for param in model_b])
            
            # Calculate cosine similarity
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            # Avoid division by zero
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return np.dot(vec_a, vec_b) / (norm_a * norm_b)
        
        elif self.similarity_metric == "euclidean":
            # Calculate negative Euclidean distance (higher means more similar)
            vec_a = np.concatenate([param.flatten() for param in model_a])
            vec_b = np.concatenate([param.flatten() for param in model_b])
            return float(-np.linalg.norm(vec_a - vec_b))
        
        elif self.similarity_metric == "layer_wise":
            # Calculate average layer-wise similarity
            similarities = []
            for layer_a, layer_b in zip(model_a, model_b):
                flat_a = layer_a.flatten()
                flat_b = layer_b.flatten()
                
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)
                
                # Avoid division by zero
                if norm_a > 0 and norm_b > 0:
                    layer_sim = np.dot(flat_a, flat_b) / (norm_a * norm_b)
                    similarities.append(layer_sim)
                else:
                    similarities.append(0.0)
                    
            return float(np.mean(similarities)) if similarities else 0.0
        else:
            # Default to dot product
            vec_a = np.concatenate([param.flatten() for param in model_a])
            vec_b = np.concatenate([param.flatten() for param in model_b])
            return np.dot(vec_a, vec_b)
    
    def _calculate_attention_weights(self, models: List[List[np.ndarray]]) -> List[float]:
        """Calculate attention weights for each client's model."""
        num_clients = len(models)
        
        if num_clients <= 1:
            return [1.0] * num_clients
        
        # Calculate similarity between each client's model and all other models
        similarity_matrix = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:  # Skip self comparisons
                    similarity_matrix[i, j] = self._calculate_similarity(models[i], models[j])
        
        # Calculate attention scores based on average similarity to other models
        attention_scores = np.mean(similarity_matrix, axis=1)
        
        # Apply temperature scaling to control distribution sharpness
        attention_scores = attention_scores / self.attention_temperature
        
        # Apply softmax to get normalized attention weights
        exp_scores = np.exp(attention_scores - np.max(attention_scores))  # Subtract max for numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Apply minimum attention threshold if needed
        if self.min_attention_value > 0:
            # Ensure minimum attention value for each client
            attention_weights = np.maximum(attention_weights, self.min_attention_value)
            # Re-normalize to sum to 1
            attention_weights = attention_weights / np.sum(attention_weights)
        
        return attention_weights.tolist()
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using attention-based weighting."""
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
        
        # Calculate attention weights for each client's model
        attention_weights = self._calculate_attention_weights(client_models)
        
        # Scale attention weights by number of examples (combine attention with example-based weighting)
        total_examples = sum(client_examples)
        if total_examples > 0:
            example_weights = [num_examples / total_examples for num_examples in client_examples]
            
            # Combine attention and example weights (element-wise product, then normalize)
            combined_weights = [a * e for a, e in zip(attention_weights, example_weights)]
            sum_combined = sum(combined_weights)
            
            if sum_combined > 0:
                final_weights = [w / sum_combined for w in combined_weights]
            else:
                final_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            # Use pure attention if no example information
            final_weights = attention_weights
        
        # Perform weighted aggregation with attention-based weights
        parameters_aggregated = ndarrays_to_parameters(aggregate_ndarrays_weighted(
            client_models,
            final_weights
        ))
        
        # Update current parameters for the next round
        self.current_parameters = parameters_to_ndarrays(parameters_aggregated)
        
        # Prepare metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics)
                for _, res in results
                if res.metrics is not None
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            # Add attention weights to metrics - convert list to string for serialization
            metrics_aggregated["attention_weights"] = str(attention_weights)
        
        return parameters_aggregated, metrics_aggregated


# Custom FedAtt Strategy with metrics printing
class CustomFedAtt(FedAtt, BaseStrategy):
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
        
        # Call parent's aggregate_fit to perform the attention-based aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Report metrics using the base class method
        self.report_metrics(server_round, metrics, "aggregate fit")
        
        # Print attention weights if available
        if "attention_weights" in metrics:
            att_weights = metrics["attention_weights"]
            # Convert string representation back to a list
            import ast
            try:
                # Handle different possible types of att_weights
                if isinstance(att_weights, str):
                    att_weights_list = ast.literal_eval(att_weights)
                elif isinstance(att_weights, bytes):
                    # If it's bytes, decode to string first
                    att_weights_list = ast.literal_eval(att_weights.decode('utf-8'))
                elif isinstance(att_weights, (list, tuple)):
                    # Already a list or tuple
                    att_weights_list = att_weights
                else:
                    # Try string conversion as a fallback
                    att_weights_list = ast.literal_eval(str(att_weights))
                
                for i, weight in enumerate(att_weights_list):
                    print(f"[Server] Round {server_round} → Client {i} attention weight: {weight:.4f}")
            except (ValueError, SyntaxError, UnicodeDecodeError):
                print(f"[Server] Round {server_round} → Could not parse attention weights: {att_weights}")
        
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
