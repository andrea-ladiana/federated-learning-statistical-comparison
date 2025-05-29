"""
Noise Injection Attack

Questo modulo implementa l'attacco di noise injection che aggiunge rumore gaussiano ai dati
per degradare la qualità del training del modello federato.
"""

import torch
import numpy as np


def apply_noise_injection(images, noise_std=0.0):
    """
    Applica noise injection aggiungendo rumore gaussiano ai dati.
    
    Args:
        images (torch.Tensor): Tensore contenente le immagini da modificare
        noise_std (float): Deviazione standard del rumore gaussiano
    
    Returns:
        torch.Tensor: Immagini con rumore aggiunto
    """
    if noise_std <= 0.0:
        return images
    
    noise = torch.randn_like(images) * noise_std
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images


def select_clients_for_noise_injection(num_clients, noise_attack_fraction):
    """
    Seleziona casualmente i client da attaccare con noise injection.
    
    Args:
        num_clients (int): Numero totale di client
        noise_attack_fraction (float): Frazione di client da attaccare (0.0-1.0)
    
    Returns:
        list: Lista degli indici dei client da attaccare
    """
    if noise_attack_fraction <= 0.0:
        return []
    
    num_attacked = max(1, int(noise_attack_fraction * num_clients))
    clients_for_noise_injection = np.random.choice(range(num_clients), size=num_attacked, replace=False)
    return clients_for_noise_injection
