"""
Data Asymmetry Attack

Questo modulo implementa l'attacco data asymmetry che crea una distribuzione asimmetrica
dei dati tra i client per simulare condizioni non-IID.
"""

import numpy as np
from torch.utils.data import Subset
from .missed_class import create_missed_class_dataset


def generate_client_sizes(total_data, num_clients, min_factor, max_factor):
    """
    Genera le dimensioni dei dataset per i client in modo asimmetrico.

    Args:
        total_data: Numero totale di dati da distribuire
        num_clients: Numero di client
        min_factor: Fattore minimo per la distribuzione uniforme
        max_factor: Fattore massimo per la distribuzione uniforme

    Returns:
        list: Lista delle dimensioni dei dataset per ogni client
    """
    if min_factor <= 0 or min_factor > max_factor:
        raise ValueError("0 < min_factor <= max_factor must hold")
    # Se min_factor = max_factor, distribuisci i dati in modo uniforme
    if min_factor == max_factor:
        base_size = total_data // num_clients
        remainder = total_data % num_clients
        return [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    
    # Genera fattori casuali uniformi nell'intervallo configurato
    factors = [np.random.uniform(min_factor, max_factor) for _ in range(num_clients)]
    total_factor = sum(factors)
    
    # Calcola le dimensioni iniziali
    client_sizes = [int(total_data * f / total_factor) for f in factors]
    
    # Verifica che tutte le dimensioni siano positive
    if any(size <= 0 for size in client_sizes):
        # Se ci sono dimensioni non positive, ridistribuisci i dati in modo uniforme
        base_size = total_data // num_clients
        remainder = total_data % num_clients
        return [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    
    # Aggiusta l'ultimo client per garantire che la somma sia pari a total_data
    current_sum = sum(client_sizes)
    if current_sum != total_data:
        client_sizes[-1] += (total_data - current_sum)
    
    return client_sizes


def create_asymmetric_datasets(dataset, client_sizes, class_removal_prob=0.0):
    """
    Crea dataset asimmetrici per i client, applicando anche missed class attack.
    
    Args:
        dataset: Dataset completo
        client_sizes (list): Lista delle dimensioni dei dataset per ogni client
        class_removal_prob (float): Probabilità che un client sia soggetto a rimozione di classe
    
    Returns:
        tuple: (client_datasets, clients_with_removed_classes) dove client_datasets è una lista
               di subset del dataset e clients_with_removed_classes è un dizionario con info sulle classi rimosse
    """
    total_data = len(dataset)
    num_clients = len(client_sizes)
    
    # Permuta casualmente gli indici del dataset completo
    indices = np.random.permutation(total_data).tolist()
    
    client_datasets = []
    clients_with_removed_classes = {}  # {client_id: excluded_class}
    
    start = 0
    for i, size in enumerate(client_sizes):
        client_indices = indices[start:start+size]
        start += size
        
        # Applica missed class attack se specificato
        if class_removal_prob > 0:
            filtered_indices, excluded_class = create_missed_class_dataset(
                dataset, client_indices, class_removal_prob)
            
            if excluded_class >= 0:
                clients_with_removed_classes[i] = excluded_class
                client_indices = filtered_indices
        
        # Crea un subset del dataset per questo client
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets, clients_with_removed_classes
