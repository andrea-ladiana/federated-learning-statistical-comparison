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

import sys
from pathlib import Path

# Add path for attacks import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "attacks"))

# Importa tutte le funzioni dai moduli degli attacchi per mantenere la compatibilità
from attacks import *
