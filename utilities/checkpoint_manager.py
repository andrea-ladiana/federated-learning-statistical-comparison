"""
Sistema di checkpoint e recovery per esperimenti di lunga durata.

Questo modulo permette di salvare lo stato degli esperimenti e riprendere
l'esecuzione in caso di interruzioni.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ExperimentStatus:
    """Stato di un singolo esperimento."""
    experiment_id: str
    config_hash: str
    completed_runs: Set[int]
    failed_runs: Set[int]
    total_runs: int
    last_updated: str
    success: bool = False
    error_message: Optional[str] = None

    def is_complete(self) -> bool:
        """Verifica se l'esperimento è completato."""
        return len(self.completed_runs) == self.total_runs

    def get_remaining_runs(self) -> Set[int]:
        """Restituisce i run ancora da completare."""
        all_runs = set(range(self.total_runs))
        return all_runs - self.completed_runs - self.failed_runs

    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario serializzabile."""
        return {
            'experiment_id': self.experiment_id,
            'config_hash': self.config_hash,
            'completed_runs': list(self.completed_runs),
            'failed_runs': list(self.failed_runs),
            'total_runs': self.total_runs,
            'last_updated': self.last_updated,
            'success': self.success,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentStatus':
        """Crea da dizionario."""
        return cls(
            experiment_id=data['experiment_id'],
            config_hash=data['config_hash'],
            completed_runs=set(data['completed_runs']),
            failed_runs=set(data['failed_runs']),
            total_runs=data['total_runs'],
            last_updated=data['last_updated'],
            success=data.get('success', False),
            error_message=data.get('error_message')
        )

class CheckpointManager:
    """Gestore dei checkpoint per esperimenti lunghi."""
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File principali
        self.status_file = self.checkpoint_dir / "experiment_status.json"
        self.config_file = self.checkpoint_dir / "configurations.json"
        self.results_backup_dir = self.checkpoint_dir / "results_backup"
        self.results_backup_dir.mkdir(exist_ok=True)
        
        # Stato corrente
        self.experiment_status: Dict[str, ExperimentStatus] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        
        # Carica stato esistente
        self.load_state()
    
    def get_config_hash(self, config: Dict[str, Any]) -> str:
        """Genera hash univoco per una configurazione."""
        # Rimuovi campi che non influenzano l'esperimento
        config_copy = config.copy()
        config_copy.pop('timestamp', None)
        
        # Ordina per consistenza
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def register_experiments(self, configs: List[Dict[str, Any]], num_runs: int):
        """Registra una lista di esperimenti da eseguire."""
        logger.info(f"Registering {len(configs)} experiments with {num_runs} runs each")
        
        for config in configs:
            experiment_id = self._get_experiment_id_from_config(config)
            config_hash = self.get_config_hash(config)
            
            # Salva configurazione
            self.configurations[experiment_id] = config
            
            # Crea o aggiorna stato esperimento
            if experiment_id not in self.experiment_status:
                self.experiment_status[experiment_id] = ExperimentStatus(
                    experiment_id=experiment_id,
                    config_hash=config_hash,
                    completed_runs=set(),
                    failed_runs=set(),
                    total_runs=num_runs,
                    last_updated=datetime.now().isoformat()
                )
            else:
                # Verifica se la configurazione è cambiata
                existing_status = self.experiment_status[experiment_id]
                if existing_status.config_hash != config_hash:
                    logger.warning(f"Configuration changed for {experiment_id}, resetting progress")
                    existing_status.config_hash = config_hash
                    existing_status.completed_runs = set()
                    existing_status.failed_runs = set()
                    existing_status.total_runs = num_runs
                    existing_status.last_updated = datetime.now().isoformat()
        
        self.save_state()
        logger.info("Experiment registration completed")
    
    def mark_run_completed(self, experiment_id: str, run_id: int, success: bool = True, 
                          error_message: Optional[str] = None):
        """Marca un run come completato."""
        if experiment_id not in self.experiment_status:
            logger.error(f"Unknown experiment: {experiment_id}")
            return
        
        status = self.experiment_status[experiment_id]
        
        if success:
            status.completed_runs.add(run_id)
            status.failed_runs.discard(run_id)  # Rimuovi dai falliti se presente
        else:
            status.failed_runs.add(run_id)
            status.completed_runs.discard(run_id)  # Rimuovi dai completati se presente
        
        status.last_updated = datetime.now().isoformat()
        status.success = status.is_complete() and len(status.failed_runs) == 0
        
        if error_message:
            status.error_message = error_message
        
        # Salva immediatamente per persistenza
        self.save_state()
        
        if status.is_complete():
            logger.info(f"Experiment {experiment_id} completed: "
                       f"{len(status.completed_runs)} successful, "
                       f"{len(status.failed_runs)} failed")
    
    def get_pending_experiments(self) -> List[Dict[str, Any]]:
        """Restituisce esperimenti non ancora completati."""
        pending = []
        
        for experiment_id, status in self.experiment_status.items():
            if not status.is_complete():
                remaining_runs = list(status.get_remaining_runs())
                if remaining_runs:
                    config = self.configurations.get(experiment_id, {})
                    pending.append({
                        'experiment_id': experiment_id,
                        'config': config,
                        'remaining_runs': remaining_runs,
                        'completed_runs': len(status.completed_runs),
                        'failed_runs': len(status.failed_runs),
                        'total_runs': status.total_runs
                    })
        
        return pending
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto del progresso."""
        total_experiments = len(self.experiment_status)
        completed_experiments = sum(1 for s in self.experiment_status.values() if s.is_complete())
        
        total_runs = sum(s.total_runs for s in self.experiment_status.values())
        completed_runs = sum(len(s.completed_runs) for s in self.experiment_status.values())
        failed_runs = sum(len(s.failed_runs) for s in self.experiment_status.values())
        
        return {
            'total_experiments': total_experiments,
            'completed_experiments': completed_experiments,
            'remaining_experiments': total_experiments - completed_experiments,
            'total_runs': total_runs,
            'completed_runs': completed_runs,
            'failed_runs': failed_runs,
            'remaining_runs': total_runs - completed_runs - failed_runs,
            'overall_progress': (completed_runs / total_runs * 100) if total_runs > 0 else 0
        }
    
    def backup_results(self, results_df, filename_suffix: str = ""):
        """Crea backup dei risultati."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"results_backup_{timestamp}{filename_suffix}.csv"
        backup_path = self.results_backup_dir / backup_filename
        
        try:
            results_df.to_csv(backup_path, index=False)
            logger.info(f"Results backed up to {backup_path}")
            
            # Mantieni solo gli ultimi 10 backup
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Failed to backup results: {e}")
    
    def _cleanup_old_backups(self, max_backups: int = 10):
        """Rimuove backup vecchi."""
        backup_files = list(self.results_backup_dir.glob("results_backup_*.csv"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_backup in backup_files[max_backups:]:
            try:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")
    
    def save_state(self):
        """Salva lo stato corrente su disco."""
        try:
            # Salva stato esperimenti
            status_data = {
                exp_id: status.to_dict() 
                for exp_id, status in self.experiment_status.items()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            # Salva configurazioni
            with open(self.config_file, 'w') as f:
                json.dump(self.configurations, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint state: {e}")
    
    def load_state(self):
        """Carica lo stato dal disco."""
        try:
            # Carica stato esperimenti
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                self.experiment_status = {
                    exp_id: ExperimentStatus.from_dict(data)
                    for exp_id, data in status_data.items()                }
                logger.info(f"Loaded {len(self.experiment_status)} experiment statuses")
            
            # Carica configurazioni
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.configurations = json.load(f)
                logger.info(f"Loaded {len(self.configurations)} configurations")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint state: {e}")
            # Inizializza stato vuoto in caso di errore
            self.experiment_status = {}
            self.configurations = {}
    
    def _get_experiment_id_from_config(self, config: Dict[str, Any]) -> str:
        """Estrae l'ID esperimento dalla configurazione."""
        # Simula il metodo get_experiment_id() di ExperimentConfig
        strategy = config.get('strategy', 'unknown')
        attack = config.get('attack', 'none')
        dataset = config.get('dataset', 'unknown')
        
        attack_str = attack
        attack_params = config.get('attack_params', {})
        if attack_params:
            # CRITICAL FIX: Sort parameters to ensure consistent ID generation
            params_str = "_".join([f"{k}{v}" for k, v in sorted(attack_params.items())])
            attack_str += f"_{params_str}"
        
        return f"{strategy}_{attack_str}_{dataset}"
    
    def reset_failed_experiments(self):
        """Reimposta gli esperimenti falliti per riprovare."""
        reset_count = 0
        for status in self.experiment_status.values():
            if status.failed_runs:
                status.failed_runs.clear()
                status.last_updated = datetime.now().isoformat()
                reset_count += 1
        
        if reset_count > 0:
            self.save_state()
            logger.info(f"Reset {reset_count} failed experiments for retry")
    
    def cleanup_completed(self):
        """Rimuove esperimenti completati dai checkpoint (opzionale)."""
        completed_ids = [
            exp_id for exp_id, status in self.experiment_status.items()
            if status.is_complete()
        ]
        
        for exp_id in completed_ids:
            del self.experiment_status[exp_id]
            self.configurations.pop(exp_id, None)
        
        if completed_ids:
            self.save_state()
            logger.info(f"Cleaned up {len(completed_ids)} completed experiments")
