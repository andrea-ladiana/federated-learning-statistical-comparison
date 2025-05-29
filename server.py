import flwr as fl
from flwr.common import ndarrays_to_parameters, Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from models import Net, CNNNet, TinyMNIST, MinimalCNN, MiniResNet20  # Importazione dei modelli dal modulo condiviso
from strategies import create_strategy  # Importa il factory per le strategie
import argparse

# Funzioni per contare i parametri del modello
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_model_parameters(model):
    """Stampa i parametri del modello, suddivisi per layer e con totale"""
    total_params = 0
    print(f"\n{'=' * 50}")
    print(f"DETTAGLIO PARAMETRI DEL MODELLO:")
    print(f"{'-' * 50}")
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name}: {param_count:,} parametri")
    print(f"{'-' * 50}")
    print(f"TOTALE: {total_params:,} parametri")
    print(f"{'=' * 50}\n")
    return total_params

# 1) Definizione modello identico al client - Shallow Network
# Nota: ora utilizziamo le classi importate dal modulo models.py

# 2) Pesi iniziali (tutti a zero) - Using MiniResNet20 Model
initial_model = MiniResNet20()  # Using MiniResNet20 for all datasets
initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
initial_parameters = ndarrays_to_parameters(initial_weights)

# main
if __name__ == "__main__":    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Avvia il server di Federated Learning")
    parser.add_argument("--rounds", type=int, default=3, help="Numero di round di federazione da eseguire")
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        choices=["MNIST", "FMNIST", "CIFAR10"],
                        help="Dataset da utilizzare (MNIST, FMNIST, CIFAR10)")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="fedavg", 
        choices=["fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam", "krum", "trimmedmean", "bulyan",
                "dasha", "depthfl", "heterofl", "fedmeta", "fedper", "fjord", "flanders", "fedopt"],
        help="Strategia di aggregazione da utilizzare"
    )
    
    # Parametri per strategie specifiche
    # FedProx
    parser.add_argument("--proximal-mu", type=float, default=0.01, 
                       help="Parametro mu per FedProx (default: 0.01)")
    
    # FedAdam
    parser.add_argument("--learning-rate", type=float, default=0.1, 
                       help="Learning rate lato server per FedAdam (default: 0.1)")
    
    # FedAvgM
    parser.add_argument("--server-momentum", type=float, default=0.9, 
                       help="Momentum lato server per FedAvgM (default: 0.9)")
    
    # Krum e Bulyan
    parser.add_argument("--num-byzantine", type=int, default=0, 
                       help="Numero atteso di client malevoli per Krum/Bulyan (default: 0)")
      # TrimmedMean
    parser.add_argument("--beta", type=float, default=0.1, 
                       help="Frazione di valori da eliminare per TrimmedMean (default: 0.1)")
    
    # DASHA baseline parameters
    parser.add_argument("--step-size", type=float, default=0.5,
                       help="Step size for DASHA algorithm (default: 0.5)")
    parser.add_argument("--compressor-coords", type=int, default=10,
                       help="Number of coordinates for DASHA compression (default: 10)")
    
    # DepthFL baseline parameters
    parser.add_argument("--alpha", type=float, default=0.75,
                       help="Alpha parameter for DepthFL (default: 0.75)")
    parser.add_argument("--tau", type=float, default=0.6,
                       help="Tau parameter for DepthFL (default: 0.6)")
    
    # FLANDERS baseline parameters
    parser.add_argument("--to-keep", type=float, default=0.6,
                       help="Fraction of clients to keep for FLANDERS (default: 0.6)")
    
    # FedOpt baseline parameters
    parser.add_argument("--fedopt-tau", type=float, default=1e-3,
                       help="Tau parameter for FedOpt (default: 1e-3)")
    parser.add_argument("--fedopt-beta1", type=float, default=0.9,
                       help="Beta1 parameter for FedOpt (default: 0.9)")    
    parser.add_argument("--fedopt-beta2", type=float, default=0.99,
                       help="Beta2 parameter for FedOpt (default: 0.99)")
    parser.add_argument("--fedopt-eta", type=float, default=1e-3,
                       help="Eta parameter for FedOpt (default: 1e-3)")
    parser.add_argument("--fedopt-eta-l", type=float, default=1e-3,
                       help="Eta_l parameter for FedOpt (default: 1e-3)")
    
    args = parser.parse_args()
    print("\nAvvio del server Flower per Federated Learning...")
    print(f"Dataset selezionato: {args.dataset}")
    print(f"Utilizzo del modello: {initial_model.__class__.__name__}")
    print(f"Strategia selezionata: {args.strategy}")
    print(f"Numero di round: {args.rounds}")
    
    # Stampa il numero di parametri del modello per riferimento
    print_model_parameters(initial_model)
    
    # Prepara parametri specifici per la strategia selezionata
    strategy_kwargs = {}
    
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
        print(f"Parametro mu per FedProx: {args.proximal_mu}")
    elif args.strategy == "fedadam":
        strategy_kwargs["server_learning_rate"] = args.learning_rate
        print(f"Learning rate per FedAdam: {args.learning_rate}")
    
    elif args.strategy == "fedavgm":
        strategy_kwargs["server_momentum"] = args.server_momentum
        print(f"Server momentum per FedAvgM: {args.server_momentum}")
    
    elif args.strategy in ["krum", "bulyan"]:
        strategy_kwargs["num_byzantine"] = args.num_byzantine
        print(f"Numero di client malevoli attesi: {args.num_byzantine}")
    
    elif args.strategy == "trimmedmean":
        strategy_kwargs["beta"] = args.beta
        print(f"Frazione di trimming: {args.beta}")
    
    # Baseline strategies parameters
    elif args.strategy == "dasha":
        strategy_kwargs["step_size"] = args.step_size
        strategy_kwargs["compressor_coords"] = args.compressor_coords
        print(f"DASHA parameters - step_size: {args.step_size}, compressor_coords: {args.compressor_coords}")
    
    elif args.strategy == "depthfl":
        strategy_kwargs["alpha"] = args.alpha
        strategy_kwargs["tau"] = args.tau
        print(f"DepthFL parameters - alpha: {args.alpha}, tau: {args.tau}")
    
    elif args.strategy == "heterofl":
        print("HeteroFL: Using default parameters")
    
    elif args.strategy == "fedmeta":
        strategy_kwargs["beta"] = args.beta  # Reusing beta for FedMeta
        print(f"FedMeta parameter - beta: {args.beta}")
    
    elif args.strategy == "fedper":
        print("FedPer: Using default parameters")
    
    elif args.strategy == "fjord":
        print("FjORD: Using default parameters")
    
    elif args.strategy == "flanders":
        strategy_kwargs["to_keep"] = args.to_keep
        print(f"FLANDERS parameter - to_keep: {args.to_keep}")
    
    elif args.strategy == "fedopt":
        strategy_kwargs["tau"] = args.fedopt_tau
        strategy_kwargs["beta_1"] = args.fedopt_beta1
        strategy_kwargs["beta_2"] = args.fedopt_beta2
        strategy_kwargs["eta"] = args.fedopt_eta
        strategy_kwargs["eta_l"] = args.fedopt_eta_l
        print(f"FedOpt parameters - tau: {args.fedopt_tau}, beta_1: {args.fedopt_beta1}, beta_2: {args.fedopt_beta2}, eta: {args.fedopt_eta}, eta_l: {args.fedopt_eta_l}")
    
    # Crea la strategia utilizzando il factory
    strategy = create_strategy(
        strategy_name=args.strategy,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        fraction_evaluate=0.5,
        min_evaluate_clients=1,
        **strategy_kwargs
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
