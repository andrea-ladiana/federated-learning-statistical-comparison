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

from torchvision import datasets, transforms # per la gestione dei dataset
from models import Net, CNNNet, TinyMNIST, MinimalCNN, MiniResNet20, get_transform_common, get_transform_cifar  # Importazione dei modelli dal modulo condiviso

# Importazione delle funzioni di attacco e configurazione
import utilities.fl_attacks
from configuration.attack_config import *
from configuration.config_manager import get_config_manager
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 1) definizione del modello locale
# - I modelli sono ora importati dal modulo models.py

# 2) caricamento dei dati
# - caricamento di MNIST
# - DataLoader: gestisce il flusso degli esempi di un dataset verso il modello --> carica i dati
def load_data(cid=None, num_clients=10, apply_attacks=False, dataset_name="MNIST"):
    """
    Carica i dati per il training e test del client.

    Args:
        cid: Client ID
        num_clients: Numero totale di client partecipanti
        apply_attacks: Se applicare gli attacchi configurati
        dataset_name: Nome del dataset ("MNIST", "FMNIST", "CIFAR10")
    """
    # Seleziona il dataset e le trasformazioni appropriate per TinyMNIST (grayscale)
    if dataset_name.upper() == "MNIST":
        # TinyMNIST needs grayscale input (1 channel)
        transform = transforms.Compose([
            transforms.Resize(28),  # Keep MNIST original size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
        ])
        trainset = datasets.MNIST(".", train=True, download=True, transform=transform)
        testset = datasets.MNIST(".", train=False, download=True, transform=transform)
    elif dataset_name.upper() == "FMNIST":
        # TinyMNIST needs grayscale input (1 channel)
        transform = transforms.Compose([
            transforms.Resize(28),  # Keep Fashion-MNIST original size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
        ])
        trainset = datasets.FashionMNIST(".", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(".", train=False, download=True, transform=transform)
    elif dataset_name.upper() == "CIFAR10":
        # CIFAR-10 is not supported with TinyMNIST model (requires grayscale input)
        raise ValueError(f"CIFAR-10 non è supportato con il modello TinyMNIST. Usa solo MNIST o Fashion-MNIST.")
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}. Supportati: MNIST, FMNIST")
    
    print(f"[Client {cid}] Caricamento dataset: {dataset_name}")
    
    # Definizione del numero di esempi per client (può essere personalizzata)
    num_train_examples = 1000
    num_test_examples = 200
    
    # Applicazione degli attacchi se richiesto
    if apply_attacks and ENABLE_ATTACKS:
        # Se il client ID è fornito, possiamo usarlo per determinare quali client attaccare
        client_id = int(cid) if cid is not None else 0
        
        # 4. Data Asymmetry (modifica il numero di esempi per client)
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
        if MISSED_CLASS["enabled"]:
            train_indices, excluded_class = utilities.fl_attacks.create_missed_class_dataset(
                trainset, train_indices, MISSED_CLASS["class_removal_prob"])
            if excluded_class >= 0:
                print(f"[Client {cid}] Missed Class Attack: classe {excluded_class} rimossa")
        
        # 5. Targeted Label-flipping (cambia le etichette da una classe sorgente a una target)
        if LABEL_FLIPPING["enabled"]:
            # Determina se questo client è un target per l'attacco
            clients_to_attack = utilities.fl_attacks.select_clients_for_label_flipping(
                num_clients, LABEL_FLIPPING["attack_fraction"])
            
            if client_id in clients_to_attack:
                # Seleziona classi sorgente e target
                source_class, target_class = utilities.fl_attacks.select_source_target_classes(
                    fixed_source=LABEL_FLIPPING["fixed_source"],
                    fixed_target=LABEL_FLIPPING["fixed_target"])
                
                # Crea un sottoinsieme del dataset per questo client
                client_subset = torch.utils.data.Subset(trainset, train_indices)
                
                # Applica il label flipping
                modified_dataset, num_flipped = utilities.fl_attacks.apply_targeted_label_flipping(
                    trainset, train_indices, source_class, target_class, 
                    LABEL_FLIPPING["flip_probability"])
                
                # Sostituisci il subset originale con quello modificato
                trainset_for_loader = modified_dataset
                print(f"[Client {cid}] Label Flipping Attack: {num_flipped} etichette cambiate "
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
    def __init__(self, cid: str, num_clients: int, dataset_name: str = "MNIST"):
        self.cid = cid
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.model = TinyMNIST()  # Using TinyMNIST for all datasets
        # Carica i dati con potenziali attacchi applicati
        self.trainloader, self.testloader = load_data(cid=cid, num_clients=num_clients, apply_attacks=ENABLE_ATTACKS, dataset_name=dataset_name)
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
        if ENABLE_ATTACKS and CLIENT_FAILURE["enabled"]:
            # Semplice check di probabilità diretto
            if np.random.rand() < CLIENT_FAILURE["failure_prob"]:
                print(f"[Client {self.cid}] SIMULAZIONE FALLIMENTO (TERMINAZIONE) DURANTE FIT - ROUND {self.current_round} - PROB {CLIENT_FAILURE['failure_prob']}")
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
            if ENABLE_ATTACKS and NOISE_INJECTION["enabled"]:
                # Verifica se questo client è un target per noise injection
                noise_clients = utilities.fl_attacks.select_clients_for_noise_injection(
                    self.num_clients, NOISE_INJECTION["attack_fraction"])
                
                if int(self.cid) in noise_clients:
                    # Applica noise injection ai dati
                    data = utilities.fl_attacks.apply_noise_injection(data, NOISE_INJECTION["noise_std"])
                    if batch_idx == 1:  # Stampa solo una volta per round
                        print(f"[Client {self.cid}] Noise Injection Attack: std={NOISE_INJECTION['noise_std']}")
            
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

        avg_loss = total_loss / total
        accuracy = correct / total
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
        if ENABLE_ATTACKS and CLIENT_FAILURE["enabled"]:
            # Semplice check di probabilità diretto
            if np.random.rand() < CLIENT_FAILURE["failure_prob"]:
                print(f"[Client {self.cid}] SIMULAZIONE FALLIMENTO (TERMINAZIONE) DURANTE EVALUATE - ROUND {self.current_round} - PROB {CLIENT_FAILURE['failure_prob']}")
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

        avg_loss = total_loss / total
        accuracy = correct / total
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
    
    # Handle global attack enablement switch
    if args.enable_attacks:
        ENABLE_ATTACKS = True 
        print(f"[Client {args.cid}] Attacchi globalmente ABILITATI via CLI (--enable-attacks).")
    else:
        if ENABLE_ATTACKS:
             print(f"[Client {args.cid}] Attacchi globalmente ABILITATI (da attack_config.py; --enable-attacks non usato).")
        else:
             print(f"[Client {args.cid}] Attacchi globalmente DISABILITATI (da attack_config.py; --enable-attacks non usato).")

    # Override CLIENT_FAILURE settings if new CLI args are provided
    if args.client_failure_active:
        CLIENT_FAILURE["enabled"] = True
        print(f"[Client {args.cid}] Client Failure: 'enabled' IMPOSTATO A TRUE via CLI (--client-failure-active).")
    
    if args.client_failure_probability is not None:
        CLIENT_FAILURE["failure_prob"] = args.client_failure_probability
        print(f"[Client {args.cid}] Client Failure: 'failure_prob' IMPOSTATA A {CLIENT_FAILURE['failure_prob']} via CLI (--client-failure-probability).")

    # Diagnostic print for the effective Client Failure configuration
    if ENABLE_ATTACKS:
        if CLIENT_FAILURE["enabled"]:
            print(f"[Client {args.cid}] Client Failure Attack CONFIGURAZIONE ATTIVA: probabilità={CLIENT_FAILURE['failure_prob']}.")
        else:
            print(f"[Client {args.cid}] Client Failure Attack CONFIGURAZIONE NON ATTIVA (CLIENT_FAILURE['enabled'] è False).")
    else:
        print(f"[Client {args.cid}] Client Failure Attack NON ATTIVO (Attacchi globalmente disabilitati tramite ENABLE_ATTACKS=False).")
    
    # Imposta un seed per la riproducibilità, ma diverso per ogni client
    seed_value = int(args.cid) + 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    print(f"[Client {args.cid}] Inizializzato con seed {seed_value}")
    
    client = LoggingClient(cid=args.cid, num_clients=args.num_clients, dataset_name=args.dataset)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
    )
