"""
Sistema di configurazione centralizzata per esperimenti di federated learning.

Gestisce parametri di sistema, configurazioni predefinite e validazione.
"""

import yaml
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Configurazione di sistema."""
    max_retries: int = 2
    retry_delay: int = 5
    process_timeout: int = 120
    port: int = 8080
    log_level: str = "INFO"
    checkpoint_backup_interval: int = 10  # Backup ogni N esperimenti
    max_backups: int = 10
    
    def validate(self):
        """Valida la configurazione di sistema."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.process_timeout <= 0:
            raise ValueError("process_timeout must be positive")
        if not (1024 <= self.port <= 65535):
            raise ValueError("port must be between 1024 and 65535")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log_level must be a valid logging level")

@dataclass
class ExperimentDefaults:
    """Configurazioni predefinite per esperimenti."""
    num_rounds: int = 1
    num_clients: int = 2
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 1
    
    def validate(self):
        """Valida le configurazioni predefinite."""
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")

@dataclass
class ResourceLimits:
    """Limiti delle risorse di sistema."""
    max_memory_mb: int = 8192  # 8GB
    max_cpu_percent: float = 80.0
    max_parallel_experiments: int = 1
    disk_space_threshold_mb: int = 1024  # 1GB spazio minimo richiesto
    
    def validate(self):
        """Valida i limiti delle risorse."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if not (0 < self.max_cpu_percent <= 100):
            raise ValueError("max_cpu_percent must be between 0 and 100")
        if self.max_parallel_experiments <= 0:
            raise ValueError("max_parallel_experiments must be positive")

class ConfigManager:
    """Gestore della configurazione centralizzata."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("experiment_config.yaml")
        self.system = SystemConfig()
        self.defaults = ExperimentDefaults()
        self.resources = ResourceLimits()
        
        # Carica configurazione esistente
        self.load_config()
        
        # Valida configurazione
        self.validate_all()
    
    def load_config(self):
        """Carica configurazione da file YAML."""
        if not self.config_file.exists():
            logger.info(f"Config file {self.config_file} not found, using defaults")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Carica configurazione di sistema
            if 'system' in config_data:
                for key, value in config_data['system'].items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)
                    else:
                        logger.warning(f"Unknown system config key: {key}")
            
            # Carica configurazione predefinita esperimenti
            if 'defaults' in config_data:
                for key, value in config_data['defaults'].items():
                    if hasattr(self.defaults, key):
                        setattr(self.defaults, key, value)
                    else:
                        logger.warning(f"Unknown defaults config key: {key}")
            
            # Carica limiti risorse
            if 'resources' in config_data:
                for key, value in config_data['resources'].items():
                    if hasattr(self.resources, key):
                        setattr(self.resources, key, value)
                    else:
                        logger.warning(f"Unknown resources config key: {key}")
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}")
            logger.info("Using default configuration")
    
    def save_config(self):
        """Salva configurazione su file YAML."""
        try:
            config_data = {
                'system': asdict(self.system),
                'defaults': asdict(self.defaults),
                'resources': asdict(self.resources)
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")
    
    def validate_all(self):
        """Valida tutta la configurazione."""
        try:
            self.system.validate()
            self.defaults.validate()
            self.resources.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_strategy_params(self, strategy: str) -> Dict[str, Any]:
        """Restituisce parametri predefiniti per una strategia."""
        strategy_params = {
            "fedprox": {"proximal_mu": 0.01},
            "fedavgm": {"server_momentum": 0.9},
            "fedadam": {"server_learning_rate": 0.1},
            "krum": {"num_byzantine": 2},
            "trimmedmean": {"beta": 0.1},
            "bulyan": {"num_byzantine": 2},
            "dasha": {"step_size": 0.5, "compressor_coords": 10},
            "depthfl": {"alpha": 0.75, "tau": 0.6},
            "flanders": {"to_keep": 0.6},
            "fedopt": {
                "tau": 1e-3,
                "beta_1": 0.9,
                "beta_2": 0.99,
                "eta": 1e-3,
                "eta_l": 1e-3,
            },
        }
        return strategy_params.get(strategy, {})
    
    def get_attack_params(self, attack: str) -> List[Dict[str, Any]]:
        """Restituisce variazioni di parametri per un attacco."""
        attack_params = {
            "none": [{}],
            "noise": [
                {"noise_std": 0.1, "noise_fraction": 0.3},
                {"noise_std": 0.5, "noise_fraction": 0.8},
            ],
            "missed": [
                {"missed_prob": 0.3},
                {"missed_prob": 0.8},
            ],
            "failure": [
                {"failure_prob": 0.3},
                {"failure_prob": 0.8},
            ],
            "asymmetry": [
                {"asymmetry_min": 0.5, "asymmetry_max": 1.5},
                {"asymmetry_min": 0.1, "asymmetry_max": 3.0},
            ],
            "labelflip": [
                {"labelflip_fraction": 0.1, "flip_prob": 0.3},
                {"labelflip_fraction": 0.5, "flip_prob": 0.8},
            ],
            "gradflip": [
                {"gradflip_fraction": 0.1, "gradflip_intensity": 0.3},
                {"gradflip_fraction": 0.5, "gradflip_intensity": 0.8},
            ],
        }
        return attack_params.get(attack, [{}])
    
    def get_valid_strategies(self) -> List[str]:
        """Restituisce lista delle strategie valide."""
        return [
            "fedavg", "fedavgm", "fedprox", "fednova", "scaffold", "fedadam",
            "krum", "trimmedmean", "bulyan",
            "dasha", "depthfl", "heterofl", "fedmeta", "fedper",
            "fjord", "flanders", "fedopt",
        ]
    
    def get_valid_attacks(self) -> List[str]:
        """Restituisce lista degli attacchi validi."""
        return ["none", "noise", "missed", "failure", "asymmetry", "labelflip", "gradflip"]
    
    def get_valid_datasets(self) -> List[str]:
        """Restituisce lista dei dataset validi."""
        return ["MNIST", "FMNIST", "CIFAR10"]
    
    def check_system_resources(self) -> Dict[str, bool]:
        """Verifica la disponibilità delle risorse di sistema."""
        import psutil
        import shutil
        
        checks = {
            "memory_available": True,
            "cpu_available": True,
            "disk_space_available": True
        }
        
        try:
            # Controlla memoria
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)
            checks["memory_available"] = available_memory_mb >= self.resources.max_memory_mb / 2
            
            # Controlla CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            checks["cpu_available"] = cpu_percent < self.resources.max_cpu_percent
            
            # Controlla spazio disco
            disk_usage = shutil.disk_usage(Path.cwd())
            free_space_mb = disk_usage.free / (1024 * 1024)
            checks["disk_space_available"] = free_space_mb >= self.resources.disk_space_threshold_mb
            
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
        
        return checks
    
    def setup_logging(self):
        """Configura il logging secondo le impostazioni."""
        log_level = getattr(logging, self.system.log_level.upper())
        
        # Rimuovi handler esistenti
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configura nuovo logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('experiment_system.log'),
                logging.StreamHandler()
            ]
        )

# Istanza globale del config manager
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Restituisce l'istanza singleton del config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def create_default_config_file():
    """Crea un file di configurazione di esempio."""
    config_path = Path("experiment_config.yaml")
    
    if config_path.exists():
        logger.info(f"Config file {config_path} already exists")
        return
    
    default_config = {
        'system': {
            'max_retries': 3,
            'retry_delay': 5,
            'process_timeout': 600,
            'port': 8080,
            'log_level': 'INFO',
            'checkpoint_backup_interval': 50,
            'max_backups': 10
        },
        'defaults': {
            'num_rounds': 10,
            'num_clients': 10,
            'learning_rate': 0.01,
            'batch_size': 32,
            'local_epochs': 1
        },
        'resources': {
            'max_memory_mb': 8192,
            'max_cpu_percent': 80.0,
            'max_parallel_experiments': 1,
            'disk_space_threshold_mb': 1024
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        logger.info(f"Created default config file: {config_path}")
    except Exception as e:
        logger.error(f"Failed to create default config file: {e}")
