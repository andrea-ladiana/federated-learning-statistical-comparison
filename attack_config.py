"""
Configurazione per gli attacchi di Federated Learning.
Questo file contiene tutti i parametri di configurazione per i vari tipi di attacchi.
"""

# Configurazione generale
ENABLE_ATTACKS = True  # Abilita/disabilita globalmente tutti gli attacchi

# 1. Noise Injection
NOISE_INJECTION = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "noise_std": 0.1,  # Deviazione standard del rumore gaussiano
    "attack_fraction": 0.2,  # Frazione di client da attaccare (0.0-1.0)
}

# 2. Missed Class
MISSED_CLASS = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "class_removal_prob": 0.3,  # Probabilità che un client sia soggetto a rimozione di classe
}

# 3. Client Failure
CLIENT_FAILURE = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "failure_prob": 0.1,  # Probabilità che un client fallisca in un round specifico
    "debug_mode": True,  # Se True, stampa informazioni aggiuntive di debug
}

# 4. Data Asymmetry
DATA_ASYMMETRY = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "min_factor": 0.5,  # Fattore minimo per la distribuzione uniforme
    "max_factor": 3.0,  # Fattore massimo per la distribuzione uniforme
    "class_removal_prob": 0.0,  # Probabilità che un client sia soggetto a rimozione di classe
}

# 5. Targeted Label-flipping Attack
LABEL_FLIPPING = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "attack_fraction": 0.2,  # Frazione di client da attaccare (0.0-1.0)
    "flip_probability": 0.8,  # Probabilità che un'etichetta della classe sorgente venga cambiata
    "fixed_source": None,  # Classe sorgente fissa (None per casuale)
    "fixed_target": None,  # Classe target fissa (None per casuale)
    "change_each_round": True,  # Se True, cambia le classi sorgente/target ad ogni round
}

# 6. Gradient Flipping Attack
GRADIENT_FLIPPING = {
    "enabled": False,  # Abilita/disabilita questo tipo di attacco
    "attack_fraction": 0.2,  # Frazione di client da attaccare (0.0-1.0)
    "flip_intensity": 1.0,  # Intensità dell'attacco (1.0 = flip completo, 0.5 = flip parziale)
}

# Registro dello stato degli attacchi (utile per tracking tra round)
ATTACK_STATE = {
    "client_failure_history": {},  # Storico dei fallimenti dei client
    "source_target_history": {},  # Storico delle classi sorgente/target per round
    "current_round": 0,  # Round corrente di federazione
    "broken_clients_in_current_round": set(),  # Set di client rotti nel round corrente
    "gradient_flipping_history": {},  # Storico dei client che hanno applicato gradient flipping
}
