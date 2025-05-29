"""
Client Failure Attack

Questo modulo implementa l'attacco client failure che simula il fallimento di un client
durante il training federato.
"""

import numpy as np


def is_client_broken(client_id, round_num, client_failure_prob, client_failure_history=None):
    """
    Verifica se un client è rotto in questo round specifico.
    Ogni round ha una probabilità indipendente di fallimento.
    
    Args:
        client_id: ID del client da verificare
        round_num: Numero del round corrente
        client_failure_prob: Probabilità di fallimento del client
        client_failure_history: Dizionario che tiene traccia dei fallimenti dei client
    
    Returns:
        bool: True se il client è rotto in questo round, False altrimenti
    """
    if client_failure_history is None:
        client_failure_history = {}
    
    # Crea una chiave univoca per memorizzare la decisione per questo client e round
    client_round_key = f"{client_id}_{round_num}"
    
    # Se abbiamo già deciso per questo client e round, riutilizza la decisione
    if client_round_key in client_failure_history:
        return client_failure_history[client_round_key]
        
    # Se la probabilità di fallimento è 0, il client non è mai rotto
    if client_failure_prob <= 0.0:
        client_failure_history[client_round_key] = False
        return False
    
    # Verifica se il client si rompe in questo round
    is_broken = np.random.rand() < client_failure_prob
    client_failure_history[client_round_key] = is_broken
    
    if is_broken:
        print(f"Client {client_id} è stato scelto per fallire nel round {round_num}")
    
    return is_broken
