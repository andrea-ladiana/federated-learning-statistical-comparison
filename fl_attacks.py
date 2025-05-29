"""
Implementazione dei vari tipi di attacchi per il Federated Learning su MNIST

Questo file ora funge da wrapper per mantenere la compatibilità con il codice esistente.
Gli attacchi sono stati spostati nella cartella attacks/ e organizzati in moduli separati:

1. Noise Injection - attacks/noise_injection.py
2. Missed Class - attacks/missed_class.py
3. Client Failure - attacks/client_failure.py
4. Data Asymmetry - attacks/data_asymmetry.py
5. Targeted Label-flipping Attack - attacks/label_flipping.py
6. Gradient Flipping Attack - attacks/gradient_flipping.py
"""

# Importa tutte le funzioni dai moduli degli attacchi per mantenere la compatibilità
from attacks import *
