# Attacks Module

Questo modulo contiene l'implementazione di vari tipi di attacchi per il Federated Learning su MNIST. Gli attacchi sono stati organizzati in moduli separati per migliorare la manutenibilità e la leggibilità del codice.

## Struttura del Modulo

### 1. Noise Injection (`noise_injection.py`)
- **Descrizione**: Aggiunge rumore gaussiano ai dati per degradare la qualità del training
- **Funzioni principali**:
  - `apply_noise_injection()`: Applica rumore gaussiano alle immagini
  - `select_clients_for_noise_injection()`: Seleziona i client da attaccare

### 2. Missed Class (`missed_class.py`)
- **Descrizione**: Rimuove esempi di una certa classe dal dataset di un client
- **Funzioni principali**:
  - `create_missed_class_dataset()`: Crea un dataset con classi mancanti

### 3. Client Failure (`client_failure.py`)
- **Descrizione**: Simula il fallimento di un client durante il training
- **Funzioni principali**:
  - `is_client_broken()`: Verifica se un client è fallito in un round specifico

### 4. Data Asymmetry (`data_asymmetry.py`)
- **Descrizione**: Distribuzione asimmetrica dei dati tra i client per simulare condizioni non-IID
- **Funzioni principali**:
  - `generate_client_sizes()`: Genera dimensioni asimmetriche per i dataset dei client
  - `create_asymmetric_datasets()`: Crea dataset asimmetrici per i client

### 5. Targeted Label-flipping Attack (`label_flipping.py`)
- **Descrizione**: Cambia etichette da una classe sorgente a una classe target
- **Funzioni principali**:
  - `select_clients_for_label_flipping()`: Seleziona i client da attaccare
  - `select_source_target_classes()`: Seleziona le classi sorgente e target
  - `apply_targeted_label_flipping()`: Applica l'attacco di label flipping
  - `perform_targeted_label_flipping()`: Esegue l'attacco completo

### 6. Gradient Flipping Attack (`gradient_flipping.py`)
- **Descrizione**: Inverte i gradienti dei client malevoli per sabotare l'apprendimento
- **Funzioni principali**:
  - `select_clients_for_gradient_flipping()`: Seleziona i client da attaccare
  - `apply_gradient_flipping()`: Applica l'inversione dei gradienti
  - `perform_gradient_flipping_attack()`: Esegue l'attacco completo
  - `get_gradient_flipping_attack_info()`: Genera informazioni per il logging

## Utilizzo

Il modulo può essere importato in due modi:

### Metodo 1: Import diretto dal modulo attacks
```python
from attacks import apply_noise_injection, select_clients_for_noise_injection
from attacks.label_flipping import perform_targeted_label_flipping
```

### Metodo 2: Import tramite fl_attacks.py (compatibilità retroattiva)
```python
import fl_attacks
# Tutte le funzioni sono disponibili come prima
```

## Compatibilità

Il file `fl_attacks.py` originale è stato modificato per mantenere la compatibilità con il codice esistente. Tutte le funzioni sono ancora disponibili attraverso l'import di `fl_attacks`, ma ora il codice è organizzato in moduli separati per una migliore manutenibilità.

## Dipendenze

- `torch`: Per le operazioni sui tensori
- `numpy`: Per le operazioni numeriche
- `torch.utils.data`: Per la gestione dei dataset

## Note

Ogni modulo è autocontenuto e può essere utilizzato indipendentemente. Le dipendenze tra moduli sono minimali e chiaramente indicate negli import.
