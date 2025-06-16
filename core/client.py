import flwr as fl  # Libreria Flower
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Torch: librerie per l'addestramento di reti neurali
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.sgd import SGD  # Explicitly import SGD from correct module
import torch.nn.functional as F
import torch.utils.data  # Add this import for Subset and DataLoader

from torchvision import datasets, transforms  # per la gestione dei dataset
from models import (
    Net,
    CNNNet,
    TinyMNIST,
    MinimalCNN,
    MiniResNet20,
    get_transform_common,
    get_transform_cifar,
)  # Importazione dei modelli dal modulo condiviso

# Importazione delle funzioni di attacco e configurazione
import utilities.fl_attacks
from configuration.attack_config import create_attack_config, AttackConfig
from configuration.config_manager import get_config_manager
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# Percorso dove salvare i dataset scaricati
DATA_DIR = Path("./dataset")
DATA_DIR.mkdir(exist_ok=True)

# Cache per evitare di riscaricare i dataset ad ogni creazione di un client
DATASET_CACHE = {}

# Funzione di supporto per ottenere il dataset, con caching
def _get_cached_dataset(dataset_name: str):
    name = dataset_name.upper()
    if name in DATASET_CACHE:
        return DATASET_CACHE[name]

    if name == "MNIST":
        transform = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        testset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    elif name == "FMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    elif name == "CIFAR10":
        raise ValueError(
            "CIFAR-10 non è supportato con il modello TinyMNIST. Usa solo MNIST o Fashion-MNIST."
        )
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}. Supportati: MNIST, FMNIST")

    DATASET_CACHE[name] = (trainset, testset)
    return trainset, testset

# 1) definizione del modello locale
# - I modelli sono ora importati dal modulo models.py

# 2) caricamento dei dati
# - caricamento di MNIST
# - DataLoader: gestisce il flusso degli esempi di un dataset verso il modello --> carica i dati
from typing import Optional


def load_data(cid=None, num_clients=10, apply_attacks=False, dataset_name="MNIST", attack_cfg: Optional[AttackConfig] = None):
    """
    Carica i dati per il training e test del client.

    Args:
        cid: Client ID
        num_clients: Numero totale di client partecipanti
        apply_attacks: Se applicare gli attacchi configurati
        dataset_name: Nome del dataset ("MNIST", "FMNIST", "CIFAR10")
    """
    # Carica dataset e trasformazioni riutilizzando la cache se disponibile
    trainset, testset = _get_cached_dataset(dataset_name)
    
    print(f"[Client {cid}] Caricamento dataset: {dataset_name}")
    
    # Definizione del numero di esempi per client (può essere personalizzata)
    num_train_examples = 1000
    num_test_examples = 200
    
    if attack_cfg is None:
        attack_cfg = create_attack_config()

    ENABLE_ATTACKS = attack_cfg.enable_attacks

    # Applicazione degli attacchi se richiesto
    if apply_attacks and ENABLE_ATTACKS:
        # Se il client ID è fornito, possiamo usarlo per determinare quali client attaccare
        client_id = int(cid) if cid is not None else 0
        
        # 4. Data Asymmetry (modifica il numero di esempi per client)
        DATA_ASYMMETRY = attack_cfg.data_asymmetry
        if DATA_ASYMMETRY["enabled"]:
            # Calcola la dimensione del dataset in base all'asimmetria
            # Usa un approccio semplice: client con id più alto ricevono più dati
            if client_id % 10 == 0:  # Esempio: modifica solo alcuni client
                factor = DATA_ASYMMETRY["max_factor"]
            else:
                factor = DATA_ASYMMETRY["min_factor"]
            
            num_train_examples = int(num_train_examples * factor)
            num_train_examples = max(50, min(5000, num_train_examples))  # Limite min/max
        
        # Prepared indices for a client
        dataset_size = len(trainset)
        replace = dataset_size < num_train_examples
        rng = np.random.default_rng(seed=client_id)
        train_indices = rng.choice(dataset_size, num_train_examples, replace=replace).tolist()
        
        # 2. Missed Class (rimuove esempi di una certa classe)
        MISSED_CLASS = attack_cfg.missed_class
        if MISSED_CLASS["enabled"]:
            train_indices, excluded_class = utilities.fl_attacks.create_missed_class_dataset(
                trainset, train_indices, MISSED_CLASS["class_removal_prob"])
            if excluded_class >= 0:
                print(f"[Client {cid}] Missed Class Attack: classe {excluded_class} rimossa")
        
        # 5. Targeted Label-flipping (cambia le etichette da una classe sorgente a una target)
        LABEL_FLIPPING = attack_cfg.label_flipping
        if LABEL_FLIPPING["enabled"]:
            # Determina se questo client è un target per l'attacco
            clients_to_attack = utilities.fl_attacks.select_clients_for_label_flipping(
                num_clients, LABEL_FLIPPING["attack_fraction"])
            
            if client_id in clients_to_attack:
                # Seleziona classi sorgente e target
                source_class, target_class = utilities.fl_attacks.select_source_target_classes(
                    fixed_source=LABEL_FLIPPING["fixed_source"],
                    fixed_target=LABEL_FLIPPING["fixed_target"])

                # Applica il label flipping al sottoinsieme del client
                modified_dataset, num_flipped = utilities.fl_attacks.apply_targeted_label_flipping(
                    trainset, train_indices, source_class, target_class,
                    LABEL_FLIPPING["flip_probability"])

                # Sostituisci il subset originale con quello modificato
                trainset_for_loader = modified_dataset
                print(
                    f"[Client {cid}] Label Flipping Attack: {num_flipped} etichette cambiate "
                    f"da classe {source_class} a classe {target_class}")
            else:
                trainset_for_loader = torch.utils.data.Subset(trainset, train_indices)
        else:
            trainset_for_loader = torch.utils.data.Subset(trainset, train_indices)
    else:
        # Dataset originale senza attacchi
        trainset_for_loader = torch.utils.data.Subset(trainset, range(num_train_examples))
    
    # Indici per il test set
    test_indices = range(num_test_examples)
    testset_for_loader = torch.utils.data.Subset(testset, test_indices)
    
    # Creazione dei data loader
    trainloader = torch.utils.data.DataLoader(
        trainset_for_loader,
        batch_size=32,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset_for_loader,
        batch_size=32,
        shuffle=False,
    )
    
    return trainloader, testloader

# 3) creazione della classe: Flower Client
# componenti principali: 
# - inizializzazione della classe 
# - get parametri
# - fase di fit (training locale) --> SGD, lr 0.01, CrossEntropyLoss
# - fase di evaluate (valutazione modello globale sui client)
class LoggingClient(fl.client.NumPyClient):
    def __init__(self, cid: str, num_clients: int, dataset_name: str = "MNIST", attack_cfg: Optional[AttackConfig] = None):
        self.cid = cid
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.attack_cfg = attack_cfg or create_attack_config()
        self.model = TinyMNIST()  # Using TinyMNIST for all datasets
        # Carica i dati con potenziali attacchi applicati
        self.trainloader, self.testloader = load_data(cid=cid, num_clients=num_clients, apply_attacks=self.attack_cfg.enable_attacks, dataset_name=dataset_name, attack_cfg=self.attack_cfg)
        self.current_round = 0
        print(f"[Client {cid}] Inizializzato con modello TinyMNIST per dataset {dataset_name}")

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters | config: {config}")
        
        # Aggiorna il round corrente se presente nella configurazione
        if "round" in config:
            current_round = config["round"]
            self.current_round = current_round
            print(f"[Client {self.cid}] Round corrente: {current_round}")
        
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print(f"[Client {self.cid}] Sending {len(params)} parameter arrays")
        return params

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit | Received parameters, config: {config}")
        
        # Aggiorna il round corrente se presente nella configurazione
        if "round" in config:
            current_round = config["round"]
            self.current_round = current_round
        
        # Implementazione semplificata del Client Failure Attack
        if self.attack_cfg.enable_attacks and self.attack_cfg.client_failure["enabled"]:
            # Semplice check di probabilità diretto
            if np.random.rand() < self.attack_cfg.client_failure["failure_prob"]:
                print(f"[Client {self.cid}] SIMULAZIONE FALLIMENTO (TERMINAZIONE) DURANTE FIT - ROUND {self.current_round} - PROB {self.attack_cfg.client_failure['failure_prob']}")
                # Fallimento immediato che termina il client
                raise SystemExit(f"Client {self.cid} failed (terminated) in round {self.current_round} (fit)")
        
        # Carica pesi dal server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        # Training locale
        print(f"[Client {self.cid}] Starting local training")
        # Use momentum=0.9 for CNN as specified in the instructions
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        for batch_idx, (data, target) in enumerate(self.trainloader, 1):
            # 1. Noise Injection Attack
            if self.attack_cfg.enable_attacks and self.attack_cfg.noise_injection["enabled"]:
                # Verifica se questo client è un target per noise injection
                noise_clients = utilities.fl_attacks.select_clients_for_noise_injection(
                    self.num_clients, self.attack_cfg.noise_injection["attack_fraction"])
                
                if int(self.cid) in noise_clients:
                    # Applica noise injection ai dati
                    data = utilities.fl_attacks.apply_noise_injection(data, self.attack_cfg.noise_injection["noise_std"])
                    if batch_idx == 1:  # Stampa solo una volta per round
                        print(f"[Client {self.cid}] Noise Injection Attack: std={self.attack_cfg.noise_injection['noise_std']}")
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if batch_idx % 10 == 0:
                print(f"[Client {self.cid}] Batch {batch_idx}/{len(self.trainloader)} | "
                      f"loss={(total_loss/total):.4f}, acc={(correct/total):.4f}")

        if total > 0:
            avg_loss = total_loss / total
            accuracy = correct / total
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            print(
                f"[Client {self.cid}] No training data available, returning null metrics"
            )
            avg_loss = accuracy = precision = recall = f1 = 0.0
        print(
            f"[Client {self.cid}] fit complete | avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )

        # Invio parametri aggiornati
        updated_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print(f"[Client {self.cid}] Uploading updated parameters")
        return updated_params, total, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate | Received parameters, config: {config}")
        
        # Aggiorna il round corrente se presente nella configurazione
        if "round" in config:
            current_round = config["round"]
            self.current_round = current_round
        
        # Implementazione semplificata del Client Failure Attack
        if self.attack_cfg.enable_attacks and self.attack_cfg.client_failure["enabled"]:
            # Semplice check di probabilità diretto
            if np.random.rand() < self.attack_cfg.client_failure["failure_prob"]:
                print(f"[Client {self.cid}] SIMULAZIONE FALLIMENTO (TERMINAZIONE) DURANTE EVALUATE - ROUND {self.current_round} - PROB {self.attack_cfg.client_failure['failure_prob']}")
                # Fallimento immediato che termina il client
                raise SystemExit(f"Client {self.cid} failed (terminated) in round {self.current_round} (evaluate)")
        
        # Carica pesi
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Valutazione locale
        print(f"[Client {self.cid}] Starting evaluation")
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.testloader, 1):
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                if batch_idx % 5 == 0:
                    print(f"[Client {self.cid}] Eval batch {batch_idx}/{len(self.testloader)} | "
                          f"loss={(total_loss/total):.4f}, acc={(correct/total):.4f}")

        if total > 0:
            avg_loss = total_loss / total
            accuracy = correct / total
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            print(
                f"[Client {self.cid}] No evaluation data available, returning null metrics"
            )
            avg_loss = accuracy = precision = recall = f1 = 0.0
        print(
            f"[Client {self.cid}] evaluate complete | avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )

        return avg_loss, total, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


# main
if __name__ == "__main__":
    # Leggi un id client da CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "FMNIST", "CIFAR10"],
                        help="Dataset da utilizzare (MNIST, FMNIST, CIFAR10)")
    config_mgr = get_config_manager()
    parser.add_argument("--num-clients", type=int, default=config_mgr.defaults.num_clients,
                        help="Numero totale di client partecipanti")
    parser.add_argument("--enable-attacks", action="store_true", help="Abilita gli attacchi globalmente.")
    
    # New arguments for specific control over Client Failure attack
    parser.add_argument("--client-failure-active", action="store_true", 
                        help="Sovrascrive attack_config.py per abilitare l'attacco Client Failure.")
    parser.add_argument("--client-failure-probability", type=float, default=None,
                        help="Sovrascrive attack_config.py per impostare la probabilità di Client Failure.")
    
    args = parser.parse_args()

    attack_cfg = create_attack_config()

    # Handle global attack enablement switch
    if args.enable_attacks:
        attack_cfg.enable_attacks = True
        print(f"[Client {args.cid}] Attacchi globalmente ABILITATI via CLI (--enable-attacks).")
    else:
        if attack_cfg.enable_attacks:
            print(f"[Client {args.cid}] Attacchi globalmente ABILITATI (da attack_config.py; --enable-attacks non usato).")
        else:
            print(f"[Client {args.cid}] Attacchi globalmente DISABILITATI (da attack_config.py; --enable-attacks non usato).")

    # Override CLIENT_FAILURE settings if new CLI args are provided
    if args.client_failure_active:
        attack_cfg.client_failure["enabled"] = True
        print(f"[Client {args.cid}] Client Failure: 'enabled' IMPOSTATO A TRUE via CLI (--client-failure-active).")

    if args.client_failure_probability is not None:
        attack_cfg.client_failure["failure_prob"] = args.client_failure_probability
        print(f"[Client {args.cid}] Client Failure: 'failure_prob' IMPOSTATA A {attack_cfg.client_failure['failure_prob']} via CLI (--client-failure-probability).")

    # Diagnostic print for the effective Client Failure configuration
    if attack_cfg.enable_attacks:
        if attack_cfg.client_failure["enabled"]:
            print(f"[Client {args.cid}] Client Failure Attack CONFIGURAZIONE ATTIVA: probabilità={attack_cfg.client_failure['failure_prob']}.")
        else:
            print(f"[Client {args.cid}] Client Failure Attack CONFIGURAZIONE NON ATTIVA (CLIENT_FAILURE['enabled'] è False).")
    else:
        print(f"[Client {args.cid}] Client Failure Attack NON ATTIVO (Attacchi globalmente disabilitati tramite enable_attacks=False).")
    
    # Imposta un seed per la riproducibilità, ma diverso per ogni client
    seed_value = int(args.cid) + 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    print(f"[Client {args.cid}] Inizializzato con seed {seed_value}")
    
    client = LoggingClient(cid=args.cid, num_clients=args.num_clients, dataset_name=args.dataset, attack_cfg=attack_cfg)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
    )
