"""
Modulo attacks - Implementazioni degli attacchi per il Federated Learning

Questo pacchetto contiene l'implementazione di vari tipi di attacchi:
1. Noise Injection - Aggiunge rumore gaussiano ai dati
2. Missed Class - Rimuove esempi di una certa classe dal dataset di un client
3. Client Failure - Simula il fallimento di un client durante il training
4. Data Asymmetry - Distribuzione asimmetrica dei dati tra i client
5. Targeted Label-flipping Attack - Cambia etichette da una classe sorgente a una classe target
6. Gradient Flipping Attack - Inverte i gradienti dei client malevoli per sabotare l'apprendimento
"""

from .noise_injection import apply_noise_injection, select_clients_for_noise_injection
from .missed_class import create_missed_class_dataset
from .client_failure import is_client_broken
from .data_asymmetry import generate_client_sizes, create_asymmetric_datasets
from .label_flipping import (
    select_clients_for_label_flipping,
    select_source_target_classes,
    apply_targeted_label_flipping,
    perform_targeted_label_flipping
)
from .gradient_flipping import (
    select_clients_for_gradient_flipping,
    apply_gradient_flipping,
    perform_gradient_flipping_attack,
    get_gradient_flipping_attack_info
)

__all__ = [
    # Noise Injection
    'apply_noise_injection',
    'select_clients_for_noise_injection',
    
    # Missed Class
    'create_missed_class_dataset',
    
    # Client Failure
    'is_client_broken',
    
    # Data Asymmetry
    'generate_client_sizes',
    'create_asymmetric_datasets',
    
    # Label Flipping
    'select_clients_for_label_flipping',
    'select_source_target_classes',
    'apply_targeted_label_flipping',
    'perform_targeted_label_flipping',
    
    # Gradient Flipping
    'select_clients_for_gradient_flipping',
    'apply_gradient_flipping',
    'perform_gradient_flipping_attack',
    'get_gradient_flipping_attack_info'
]
