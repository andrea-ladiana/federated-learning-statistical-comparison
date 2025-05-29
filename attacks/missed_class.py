"""
Missed Class Attack

Questo modulo implementa l'attacco missed class che rimuove esempi di una certa classe
dal dataset di un client per simulare distribuzione non-IID estrema.
"""

import numpy as np


def create_missed_class_dataset(dataset, client_indices, class_removal_prob=0.0):
    """
    Crea un dataset in cui alcune classi sono mancanti per certi client.
    
    Args:
        dataset: Dataset originale
        client_indices (list): Indici degli esempi assegnati al client
        class_removal_prob (float): Probabilità che il client sia soggetto a rimozione di classe
    
    Returns:
        tuple: (filtered_indices, excluded_class) dove filtered_indices sono gli indici
               dopo aver rimosso la classe e excluded_class è la classe rimossa (-1 se nessuna)
    """
    # Verifica se il client viene attaccato
    if np.random.rand() >= class_removal_prob:
        return client_indices, -1
    
    excluded_class = np.random.randint(0, 10)  # Classe da escludere (da 0 a 9 per MNIST)
    # Filtra gli indici per rimuovere gli esempi con l'etichetta "excluded_class"
    filtered_indices = [idx for idx in client_indices if dataset.targets[idx] != excluded_class]
    
    return filtered_indices, excluded_class
