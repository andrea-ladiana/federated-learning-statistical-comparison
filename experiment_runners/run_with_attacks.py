"""
Script di utilità per avviare esperimenti di Federated Learning con attacchi.
Questo script avvia il server e più client con diversi attacchi configurati.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Add path to configuration directory
parent_dir = Path(__file__).parent.parent
config_dir = parent_dir / "configuration"
sys.path.insert(0, str(config_dir))

# Disabilita i messaggi di warning oneDNN di TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Riduce ulteriormente i messaggi di log di TensorFlow

# Import attack configurations with explicit path handling
import importlib.util

def load_attack_config():
    """Load attack configuration from file."""
    config_file = config_dir / "attack_config.py"
    spec = importlib.util.spec_from_file_location("attack_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {config_file}")
    
    attack_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attack_config)
    return attack_config

try:
    attack_config = load_attack_config()
    # Import the specific configurations
    NOISE_INJECTION = attack_config.NOISE_INJECTION
    MISSED_CLASS = attack_config.MISSED_CLASS
    CLIENT_FAILURE = attack_config.CLIENT_FAILURE
    DATA_ASYMMETRY = attack_config.DATA_ASYMMETRY
    LABEL_FLIPPING = attack_config.LABEL_FLIPPING
    GRADIENT_FLIPPING = attack_config.GRADIENT_FLIPPING
except Exception as e:
    print(f"Warning: Could not import attack config: {e}")
    # Fallback: define configurations locally
    NOISE_INJECTION = {
        "enabled": False,
        "noise_std": 0.1,
        "attack_fraction": 0.2,
    }
    MISSED_CLASS = {
        "enabled": False,
        "class_removal_prob": 0.3,
    }
    CLIENT_FAILURE = {
        "enabled": False,
        "failure_prob": 0.1,
        "debug_mode": True,
    }
    DATA_ASYMMETRY = {
        "enabled": False,
        "min_factor": 0.5,
        "max_factor": 3.0,
        "class_removal_prob": 0.0,
    }
    LABEL_FLIPPING = {
        "enabled": False,
        "attack_fraction": 0.2,
        "flip_probability": 0.8,
        "fixed_source": None,
        "fixed_target": None,
        "change_each_round": True,
    }
    GRADIENT_FLIPPING = {
        "enabled": False,
        "attack_fraction": 0.2,
        "flip_intensity": 1.0,
    }

def main():
    parser = argparse.ArgumentParser(description="Avvia esperimenti di FL con attacchi")
    parser.add_argument("--num-clients", type=int, default=10, help="Numero di client da avviare")
    parser.add_argument("--attack", type=str, choices=["noise", "missed", "failure", "asymmetry", "labelflip", "gradflip", "all", "none"],
                        default="none", help="Tipo di attacco da abilitare")
    
    # Parametri per Noise Injection
    parser.add_argument("--noise-std", type=float, default=0.1, 
                       help="Deviazione standard del rumore gaussiano (per Noise Injection)")
    parser.add_argument("--noise-fraction", type=float, default=0.3, 
                       help="Frazione di client da attaccare con Noise Injection (0.0-1.0)")
    
    # Parametri per Missed Class
    parser.add_argument("--missed-prob", type=float, default=0.3, 
                       help="Probabilità che un client sia soggetto a rimozione di classe (per Missed Class)")
    
    # Parametri per Client Failure
    parser.add_argument("--failure-prob", type=float, default=0.2, 
                       help="Probabilità che un client fallisca in un round specifico (per Client Failure)")
    
    # Parametri per Data Asymmetry
    parser.add_argument("--asymmetry-min", type=float, default=0.5, 
                       help="Fattore minimo per la distribuzione asimmetrica (per Data Asymmetry)")
    parser.add_argument("--asymmetry-max", type=float, default=3.0, 
                       help="Fattore massimo per la distribuzione asimmetrica (per Data Asymmetry)")
      # Parametri per Label Flipping
    parser.add_argument("--labelflip-fraction", type=float, default=0.2, 
                       help="Frazione di client da attaccare con Label Flipping (0.0-1.0)")
    parser.add_argument("--flip-prob", type=float, default=0.8, 
                       help="Probabilità che un'etichetta venga cambiata (per Label Flipping)")
    parser.add_argument("--source-class", type=int, default=None, 
                       help="Classe sorgente fissa per Label Flipping (None per casuale)")
    parser.add_argument("--target-class", type=int, default=None, 
                       help="Classe target fissa per Label Flipping (None per casuale)")
    
    # Parametri per Gradient Flipping
    parser.add_argument("--gradflip-fraction", type=float, default=0.2, 
                       help="Frazione di client da attaccare con Gradient Flipping (0.0-1.0)")
    parser.add_argument("--gradflip-intensity", type=float, default=1.0, 
                       help="Intensità dell'inversione dei gradienti (0.0-1.0, 1.0=inversione completa)")
      # Parametri aggiuntivi
    parser.add_argument("--rounds", type=int, default=10, 
                       help="Numero di round di federazione da eseguire")
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        choices=["MNIST", "FMNIST", "CIFAR10"],
                        help="Dataset da utilizzare (MNIST, FMNIST, CIFAR10)")
      # Strategia di aggregazione
    parser.add_argument("--strategy", type=str, default="fedavg", 
                       choices=["fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam", "krum", "trimmedmean", "bulyan",
                               "dasha", "depthfl", "heterofl", "fedmeta", "fedper", "fjord", "flanders", "fedopt"],
                       help="Strategia di aggregazione da utilizzare")
    
    # Parametri specifici per le strategie
    # FedProx
    parser.add_argument("--proximal-mu", type=float, default=0.01, 
                       help="Parametro mu per FedProx (default: 0.01)")
      # FedAdam
    parser.add_argument("--learning-rate", type=float, default=0.1, 
                       help="Learning rate lato server per FedAdam (default: 0.1)")
    parser.add_argument("--server-learning-rate", type=float, default=0.1, 
                       help="Server learning rate (alias for --learning-rate)")
    
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
    parser.add_argument("--fedopt-eta-l", type=float, default=1e-3,                       help="Eta_l parameter for FedOpt (default: 1e-3)")
    
    args = parser.parse_args()
      # Handle server learning rate parameter compatibility
    # If --server-learning-rate is provided, use it; otherwise use --learning-rate
    # This ensures compatibility with both parameter names
    effective_learning_rate = args.learning_rate
    if hasattr(args, 'server_learning_rate') and args.server_learning_rate != 0.1:
        effective_learning_rate = args.server_learning_rate
    args.learning_rate = effective_learning_rate
    
    # Configura gli attacchi in base all'argomento
    configure_attacks(args)    # Avvia il server in un processo separato
    print("Avvio del server...")
    
    # Determine the correct path to core/server.py based on current working directory
    current_dir = Path.cwd()
    if current_dir.name == "experiment_runners":
        # We're in the experiment_runners directory, need to go up one level
        server_path = current_dir.parent / "core" / "server.py"
    else:
        # We're in the main directory
        server_path = current_dir / "core" / "server.py"
    
    server_cmd = [sys.executable, str(server_path)]
      # Aggiungi round e strategia al comando server
    if args.rounds != 10:  # se diverso dal default
        server_cmd.extend(["--rounds", str(args.rounds)])
    
    if args.dataset != "MNIST":  # se diverso dal default
        server_cmd.extend(["--dataset", args.dataset])
    
    if args.strategy != "fedavg":  # se diverso dal default
        server_cmd.extend(["--strategy", args.strategy])
        
        # Aggiungi i parametri specifici per ogni strategia
        if args.strategy == "fedprox" and args.proximal_mu != 0.01:
            server_cmd.extend(["--proximal-mu", str(args.proximal_mu)])
            
        elif args.strategy == "fedadam" and args.learning_rate != 0.1:
            server_cmd.extend(["--learning-rate", str(args.learning_rate)])
        elif args.strategy == "fedavgm" and args.server_momentum != 0.9:
            server_cmd.extend(["--server-momentum", str(args.server_momentum)])
            
        elif args.strategy in ["krum", "bulyan"] and args.num_byzantine != 0:
            server_cmd.extend(["--num-byzantine", str(args.num_byzantine)])
            
        elif args.strategy == "trimmedmean" and args.beta != 0.1:
            server_cmd.extend(["--beta", str(args.beta)])
            
        # Baseline strategies parameters
        elif args.strategy == "dasha":
            if args.step_size != 0.5:
                server_cmd.extend(["--step-size", str(args.step_size)])
            if args.compressor_coords != 10:
                server_cmd.extend(["--compressor-coords", str(args.compressor_coords)])
                
        elif args.strategy == "depthfl":
            if args.alpha != 0.75:
                server_cmd.extend(["--alpha", str(args.alpha)])
            if args.tau != 0.6:
                server_cmd.extend(["--tau", str(args.tau)])
                
        elif args.strategy == "heterofl":
            pass  # No specific parameters
            
        elif args.strategy == "fedmeta":
            if args.beta != 0.1:  # Reusing beta parameter
                server_cmd.extend(["--beta", str(args.beta)])
                
        elif args.strategy == "fedper":
            pass  # No specific parameters
            
        elif args.strategy == "fjord":
            pass  # No specific parameters
            
        elif args.strategy == "flanders":
            if args.to_keep != 0.6:
                server_cmd.extend(["--to-keep", str(args.to_keep)])
                
        elif args.strategy == "fedopt":
            if args.fedopt_tau != 1e-3:
                server_cmd.extend(["--fedopt-tau", str(args.fedopt_tau)])
            if args.fedopt_beta1 != 0.9:
                server_cmd.extend(["--fedopt-beta1", str(args.fedopt_beta1)])
            if args.fedopt_beta2 != 0.99:
                server_cmd.extend(["--fedopt-beta2", str(args.fedopt_beta2)])
            if args.fedopt_eta != 1e-3:
                server_cmd.extend(["--fedopt-eta", str(args.fedopt_eta)])
            if args.fedopt_eta_l != 1e-3:
                server_cmd.extend(["--fedopt-eta-l", str(args.fedopt_eta_l)])
    
    print(f"Comando server: {' '.join(server_cmd)}")  # Log del comando per debug
    server_process = subprocess.Popen(server_cmd)
    
    # Attendi che il server si avvii
    print("Attesa avvio server (5 secondi)...")
    time.sleep(5)    # Avvia i client
    client_processes = []
    
    # Determine the correct path to core/client.py based on current working directory
    current_dir = Path.cwd()
    if current_dir.name == "experiment_runners":
        # We're in the experiment_runners directory, need to go up one level
        client_path = current_dir.parent / "core" / "client.py"
    else:
        # We're in the main directory
        client_path = current_dir / "core" / "client.py"
    
    for i in range(args.num_clients):
        cmd = [sys.executable, str(client_path), "--cid", str(i)]
        
        # Aggiungi il dataset se diverso dal default
        if args.dataset != "MNIST":
            cmd.extend(["--dataset", args.dataset])
        
        # Passa il flag generale --enable-attacks se qualsiasi attacco è attivo
        if args.attack != "none":
            cmd.append("--enable-attacks")
            
        # Passa i flag specifici per il Client Failure Attack se è l'attacco selezionato
        # o se "all" attacchi sono selezionati.
        # CLIENT_FAILURE["enabled"] è già stato impostato da configure_attacks
        if CLIENT_FAILURE["enabled"] and (args.attack == "failure" or args.attack == "all"):
            cmd.append("--client-failure-active")
            cmd.extend(["--client-failure-probability", str(args.failure_prob)])
            
        print(f"Avvio client {i} con comando: {' '.join(cmd)}") # Log del comando per debug
        client_process = subprocess.Popen(cmd)
        client_processes.append(client_process)
        # Breve pausa tra l'avvio dei client
        time.sleep(0.5)
    
    # Attendi il completamento dei processi client
    print("Tutti i client sono stati avviati. In attesa del completamento...")
    for i, process in enumerate(client_processes):
        process.wait()
        print(f"Client {i} terminato")
    
    # Termina il server
    print("Terminazione del server...")
    server_process.terminate()
    server_process.wait()
    
    print("Esperimento completato")

def configure_attacks(args):
    """Configura i parametri degli attacchi in base agli argomenti specificati."""    # Reset configuration
    NOISE_INJECTION["enabled"] = False
    MISSED_CLASS["enabled"] = False
    CLIENT_FAILURE["enabled"] = False
    DATA_ASYMMETRY["enabled"] = False
    LABEL_FLIPPING["enabled"] = False
    GRADIENT_FLIPPING["enabled"] = False
    
    attack_type = args.attack
    
    if attack_type == "none":
        print("Nessun attacco abilitato")
        return
    
    if attack_type == "noise" or attack_type == "all":
        NOISE_INJECTION["enabled"] = True
        NOISE_INJECTION["noise_std"] = args.noise_std
        NOISE_INJECTION["attack_fraction"] = args.noise_fraction
        print(f"Attacco Noise Injection abilitato: std={args.noise_std}, fraction={args.noise_fraction}")
    
    if attack_type == "missed" or attack_type == "all":
        MISSED_CLASS["enabled"] = True
        MISSED_CLASS["class_removal_prob"] = args.missed_prob
        print(f"Attacco Missed Class abilitato: prob={args.missed_prob}")
    
    if attack_type == "failure" or attack_type == "all":
        CLIENT_FAILURE["enabled"] = True
        CLIENT_FAILURE["failure_prob"] = args.failure_prob
        print(f"Attacco Client Failure abilitato: prob={args.failure_prob}")
    
    if attack_type == "asymmetry" or attack_type == "all":
        DATA_ASYMMETRY["enabled"] = True
        DATA_ASYMMETRY["min_factor"] = args.asymmetry_min
        DATA_ASYMMETRY["max_factor"] = args.asymmetry_max
        print(f"Attacco Data Asymmetry abilitato: min={args.asymmetry_min}, max={args.asymmetry_max}")
    
    if attack_type == "labelflip" or attack_type == "all":
        LABEL_FLIPPING["enabled"] = True
        LABEL_FLIPPING["attack_fraction"] = args.labelflip_fraction
        LABEL_FLIPPING["flip_probability"] = args.flip_prob
        LABEL_FLIPPING["fixed_source"] = args.source_class
        LABEL_FLIPPING["fixed_target"] = args.target_class
        print(f"Attacco Label Flipping abilitato: fraction={args.labelflip_fraction}, "
              f"flip_prob={args.flip_prob}, source={args.source_class}, target={args.target_class}")
    
    if attack_type == "gradflip" or attack_type == "all":
        GRADIENT_FLIPPING["enabled"] = True
        GRADIENT_FLIPPING["attack_fraction"] = args.gradflip_fraction
        GRADIENT_FLIPPING["flip_intensity"] = args.gradflip_intensity
        print(f"Attacco Gradient Flipping abilitato: fraction={args.gradflip_fraction}, "
              f"intensity={args.gradflip_intensity}")
    
    if attack_type == "all":
        print("Tutti gli attacchi sono abilitati con i parametri specificati")

if __name__ == "__main__":
    main()
