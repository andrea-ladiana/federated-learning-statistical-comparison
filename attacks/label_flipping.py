"""
Targeted Label-flipping Attack

Questo modulo implementa l'attacco di label flipping mirato che cambia etichette
da una classe sorgente a una classe target per degradare le prestazioni del modello.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset


def select_clients_for_label_flipping(num_clients, label_flipping_fraction):
    """
    Seleziona casualmente i client da attaccare con label flipping.
    
    Args:
        num_clients (int): Numero totale di client
        label_flipping_fraction (float): Frazione di client da attaccare (0.0-1.0)
    
    Returns:
        list: Lista degli indici dei client da attaccare
    """
    if label_flipping_fraction <= 0.0:
        return []
    
    num_attacked = max(1, int(label_flipping_fraction * num_clients))
    clients_for_label_flipping = np.random.choice(range(num_clients), size=num_attacked, replace=False)
    return clients_for_label_flipping


def select_source_target_classes(num_classes=10, fixed_source=None, fixed_target=None):
    """
    Seleziona casualmente una classe sorgente e una classe target per il label flipping.
    
    Args:
        num_classes (int): Numero totale di classi (default: 10 per MNIST)
        fixed_source (int, optional): Se specificato, usa questa classe come sorgente invece di scegliere casualmente
        fixed_target (int, optional): Se specificato, usa questa classe come target invece di scegliere casualmente
    
    Returns:
        tuple: (source_class, target_class) le classi sorgente e target selezionate
    """
    # Scegli la classe sorgente
    if fixed_source is not None:
        source_class = fixed_source
    else:
        source_class = np.random.randint(0, num_classes)
    
    # Scegli la classe target (diversa dalla sorgente)
    if fixed_target is not None:
        target_class = fixed_target
    else:
        target_class = np.random.randint(0, num_classes)
        while target_class == source_class:
            target_class = np.random.randint(0, num_classes)
    
    return source_class, target_class


def apply_targeted_label_flipping(dataset, client_indices, source_class, target_class, flip_probability=0.5):
    """
    Applica l'attacco di label flipping mirato a un subset specifico del dataset.
    
    Args:
        dataset: Dataset originale
        client_indices (list): Indici degli esempi assegnati al client
        source_class (int): Classe sorgente le cui etichette verranno cambiate
        target_class (int): Classe target in cui saranno cambiate le etichette
        flip_probability (float): Probabilità che un'etichetta della classe sorgente venga cambiata
    
    Returns:
        tuple: (modified_dataset, num_flipped) dove modified_dataset è il dataset modificato
               e num_flipped è il numero di etichette modificate
    """
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must be between 0 and 1")

    # Crea una copia del dataset per non modificare l'originale
    modified_targets = dataset.targets.clone() if isinstance(dataset.targets, torch.Tensor) else dataset.targets.copy()
    
    # Trova tutti gli indici del client che hanno la classe sorgente
    source_indices = [idx for idx in client_indices if dataset.targets[idx] == source_class]
    
    # Determina quali etichette verranno cambiate in base alla probabilità
    num_to_flip = int(len(source_indices) * flip_probability)
    if num_to_flip == 0 and len(source_indices) > 0:
        num_to_flip = 1  # Assicurati di cambiare almeno un'etichetta se ce ne sono di disponibili
    
    # Seleziona casualmente gli indici da modificare
    if num_to_flip > 0:
        indices_to_flip = np.random.choice(source_indices, size=min(num_to_flip, len(source_indices)), replace=False)
        
        # Applica il cambio di etichetta
        for idx in indices_to_flip:
            modified_targets[idx] = target_class
    else:
        indices_to_flip = []
    
    # Crea un nuovo dataset con le nuove etichette
    modified_dataset = TensorDataset(
        dataset.data[client_indices] if hasattr(dataset, 'data') else dataset.tensors[0][client_indices],
        modified_targets[client_indices]
    )
    
    return modified_dataset, len(indices_to_flip)


def perform_targeted_label_flipping(clients_datasets, client_indices, dataset, label_flipping_fraction, 
                                   flip_probability=0.5, fixed_source=None, fixed_target=None, 
                                   change_each_round=True, current_round=0, 
                                   source_target_history=None):
    """
    Esegue l'attacco di label flipping mirato su una selezione di client.
    
    Args:
        clients_datasets (list): Lista di dataset dei client
        client_indices (list): Indici degli esempi per ciascun client
        dataset: Dataset originale completo
        label_flipping_fraction (float): Frazione di client da attaccare
        flip_probability (float): Probabilità che un'etichetta della classe sorgente venga cambiata
        fixed_source (int, optional): Se specificato, usa questa classe come sorgente fissa
        fixed_target (int, optional): Se specificato, usa questa classe come target fisso
        change_each_round (bool): Se True, cambia le classi sorgente/target ad ogni round
        current_round (int): Numero del round corrente
        source_target_history (dict, optional): Dizionario per tenere traccia delle classi sorgente/target per round
    
    Returns:
        tuple: (modified_clients_datasets, attack_info) dove modified_clients_datasets sono i dataset modificati
               e attack_info contiene informazioni sull'attacco eseguito
    """
    if source_target_history is None:
        source_target_history = {}
    
    num_clients = len(clients_datasets)
    attack_info = {
        "attacked_clients": [],
        "source_class": -1,
        "target_class": -1,
        "labels_flipped": 0
    }
    
    # Se non ci sono client da attaccare, restituisci i dataset originali
    if label_flipping_fraction <= 0.0:
        return clients_datasets, attack_info
    
    # Seleziona i client da attaccare
    clients_to_attack = select_clients_for_label_flipping(num_clients, label_flipping_fraction)
    
    if not clients_to_attack:
        return clients_datasets, attack_info
    
    # Determina le classi sorgente e target per questo round
    if change_each_round or current_round not in source_target_history:
        source_class, target_class = select_source_target_classes(
            fixed_source=fixed_source, fixed_target=fixed_target)
        source_target_history[current_round] = (source_class, target_class)
    else:
        source_class, target_class = source_target_history[current_round]
    
    attack_info["source_class"] = source_class
    attack_info["target_class"] = target_class
    attack_info["attacked_clients"] = clients_to_attack
    
    # Crea una copia dei dataset dei client per non modificare gli originali
    modified_clients_datasets = clients_datasets.copy()
    
    # Applica l'attacco ai client selezionati
    total_flipped = 0
    for client_idx in clients_to_attack:
        # Ottieni gli indici degli esempi per questo client
        client_data_indices = client_indices[client_idx]
        
        # Applica il label flipping
        modified_dataset, num_flipped = apply_targeted_label_flipping(
            dataset, client_data_indices, source_class, target_class, flip_probability)
        
        # Aggiorna il dataset del client
        modified_clients_datasets[client_idx] = modified_dataset
        total_flipped += num_flipped
    
    attack_info["labels_flipped"] = total_flipped
    
    return modified_clients_datasets, attack_info
