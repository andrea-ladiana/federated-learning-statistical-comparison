"""
Gradient Flipping Attack

Questo modulo implementa l'attacco gradient flipping che inverte i gradienti
dei client malevoli per sabotare l'apprendimento federato.
"""

import numpy as np

MAX_FLIP_INTENSITY = 10.0


def select_clients_for_gradient_flipping(num_clients, gradient_flipping_fraction):
    """
    Seleziona casualmente i client da attaccare con gradient flipping.
    
    Args:
        num_clients (int): Numero totale di client
        gradient_flipping_fraction (float): Frazione di client da attaccare (0.0-1.0)
    
    Returns:
        list: Lista degli indici dei client da attaccare
    """
    if not 0.0 <= gradient_flipping_fraction <= 1.0:
        raise ValueError("gradient_flipping_fraction must be between 0 and 1")
    if gradient_flipping_fraction == 0.0:
        return []
    
    num_attacked = max(1, int(gradient_flipping_fraction * num_clients))
    clients_for_gradient_flipping = np.random.choice(range(num_clients), size=num_attacked, replace=False)
    return clients_for_gradient_flipping


def apply_gradient_flipping(model_parameters, flip_intensity=1.0):
    """
    Applica gradient flipping ai parametri del modello.
    
    I gradienti vengono invertiti (moltiplicati per -flip_intensity) per sabotare l'apprendimento federato.
    Questo simula un attacco malevolo dove un client invia aggiornamenti opposti alla direzione ottimale.
    
    Args:
        model_parameters (list): Lista di array numpy contenenti i parametri del modello
        flip_intensity (float): Intensità dell'attacco (1.0 = flip completo, 0.5 = flip parziale)
    
    Returns:
        list: Parametri del modello con gradient flipping applicato
    """
    if flip_intensity < 0.0 or flip_intensity > MAX_FLIP_INTENSITY:
        raise ValueError(
            f"flip_intensity must be between 0 and {MAX_FLIP_INTENSITY}"
        )
    if flip_intensity == 0.0:
        return model_parameters
    
    flipped_parameters = []
    for param in model_parameters:
        # Applica il flipping moltiplicando per -flip_intensity
        flipped_param = param * (-flip_intensity)
        flipped_parameters.append(flipped_param)
    
    return flipped_parameters


def perform_gradient_flipping_attack(original_parameters, client_id, num_clients, 
                                   gradient_flipping_fraction, flip_intensity=1.0,
                                   gradient_flipping_history=None, current_round=0):
    """
    Determina se un client deve applicare gradient flipping e lo applica se necessario.
    
    Args:
        original_parameters (list): Parametri originali del modello
        client_id (int): ID del client corrente
        num_clients (int): Numero totale di client
        gradient_flipping_fraction (float): Frazione di client da attaccare
        flip_intensity (float): Intensità dell'attacco
        gradient_flipping_history (dict, optional): Dizionario per tenere traccia degli attacchi
        current_round (int): Round corrente
    
    Returns:
        tuple: (parameters, is_attacked) dove parameters sono i parametri (modificati o originali)
               e is_attacked indica se l'attacco è stato applicato
    """
    if gradient_flipping_history is None:
        gradient_flipping_history = {}
    
    # Crea una chiave unica per questo round
    round_key = f"round_{current_round}"
      # Se non abbiamo ancora selezionato i client per questo round, fallo ora
    if round_key not in gradient_flipping_history:
        clients_to_attack = select_clients_for_gradient_flipping(num_clients, gradient_flipping_fraction)
        gradient_flipping_history[round_key] = list(clients_to_attack)
        
        if len(clients_to_attack) > 0:
            print(f"Round {current_round}: Client selezionati per gradient flipping attack: {clients_to_attack}")
    else:
        clients_to_attack = gradient_flipping_history[round_key]
    
    if client_id in clients_to_attack:
        # Applica gradient flipping
        flipped_parameters = apply_gradient_flipping(original_parameters, flip_intensity)
        print(f"Client {client_id}: Gradient flipping applicato con intensità {flip_intensity}")
        return flipped_parameters, True
    else:
        # Restituisci parametri originali
        return original_parameters, False


def get_gradient_flipping_attack_info(gradient_flipping_fraction, num_clients, flip_intensity, current_round):
    """
    Genera informazioni sull'attacco gradient flipping per il logging.
    
    Args:
        gradient_flipping_fraction (float): Frazione di client attaccati
        num_clients (int): Numero totale di client
        flip_intensity (float): Intensità dell'attacco
        current_round (int): Round corrente
    
    Returns:
        dict: Informazioni sull'attacco
    """
    num_attacked = max(1, int(gradient_flipping_fraction * num_clients)) if gradient_flipping_fraction > 0 else 0
    
    return {
        "attack_type": "gradient_flipping",
        "round": current_round,
        "total_clients": num_clients,
        "attacked_clients_count": num_attacked,
        "attack_fraction": gradient_flipping_fraction,
        "flip_intensity": flip_intensity,
        "enabled": gradient_flipping_fraction > 0.0
    }
