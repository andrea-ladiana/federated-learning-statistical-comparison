"""
Sistema di retry e recovery per esperimenti falliti.

Gestisce automaticamente i tentativi ripetuti per esperimenti che falliscono
a causa di problemi temporanei.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Tipi di fallimento degli esperimenti."""
    TIMEOUT = "timeout"
    PROCESS_ERROR = "process_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RetryConfig:
    """Configurazione per il sistema di retry."""
    max_retries: int = 3
    base_delay: float = 5.0  # secondi
    max_delay: float = 300.0  # 5 minuti
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    retry_on_types: set = None  # Se None, riprova su tutti i tipi
    
    def __post_init__(self):
        if self.retry_on_types is None:
            self.retry_on_types = {
                FailureType.TIMEOUT,
                FailureType.RESOURCE_ERROR,
                FailureType.NETWORK_ERROR,
                FailureType.UNKNOWN_ERROR
            }

class RetryManager:
    """Gestore dei retry per esperimenti."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.failure_history: Dict[str, list] = {}
    
    def classify_failure(self, error_message: str, return_code: Optional[int] = None) -> FailureType:
        """Classifica il tipo di fallimento basandosi sull'errore."""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return FailureType.TIMEOUT
        elif "permission denied" in error_lower or "access denied" in error_lower:
            return FailureType.PERMISSION_ERROR
        elif "connection" in error_lower or "network" in error_lower:
            return FailureType.NETWORK_ERROR
        elif "memory" in error_lower or "resource" in error_lower or "port" in error_lower:
            return FailureType.RESOURCE_ERROR
        elif return_code and return_code != 0:
            return FailureType.PROCESS_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def should_retry(self, experiment_id: str, failure_type: FailureType, 
                    attempt: int) -> bool:
        """Determina se un esperimento dovrebbe essere riprovato."""
        # Controlla se abbiamo superato il numero massimo di tentativi
        if attempt >= self.config.max_retries:
            return False
        
        # Controlla se il tipo di fallimento è tra quelli per cui riprovare
        if failure_type not in self.config.retry_on_types:
            logger.info(f"Not retrying {experiment_id}: failure type {failure_type} not in retry list")
            return False
        
        return True
    
    def get_delay(self, attempt: int) -> float:
        """Calcola il delay prima del prossimo tentativo."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)
    
    def record_failure(self, experiment_id: str, failure_type: FailureType, 
                      error_message: str, attempt: int):
        """Registra un fallimento nella storia."""
        if experiment_id not in self.failure_history:
            self.failure_history[experiment_id] = []
        
        self.failure_history[experiment_id].append({
            'timestamp': datetime.now().isoformat(),
            'attempt': attempt,
            'failure_type': failure_type.value,
            'error_message': error_message
        })
    
    def execute_with_retry(self, experiment_id: str, experiment_func: Callable[[], Tuple[bool, str]], 
                          *args, **kwargs) -> Tuple[bool, str]:
        """Esegue una funzione con retry automatico."""
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(f"Executing {experiment_id}, attempt {attempt + 1}/{self.config.max_retries + 1}")
                
                success, output = experiment_func(*args, **kwargs)
                
                if success:
                    if attempt > 0:
                        logger.info(f"Experiment {experiment_id} succeeded after {attempt + 1} attempts")
                    return True, output
                
                # Fallimento - determina se riprovare
                failure_type = self.classify_failure(output)
                self.record_failure(experiment_id, failure_type, output, attempt)
                
                if not self.should_retry(experiment_id, failure_type, attempt):
                    logger.error(f"Experiment {experiment_id} failed permanently after {attempt + 1} attempts")
                    return False, output
                
                # Calcola delay e attendi prima del prossimo tentativo
                if attempt < self.config.max_retries:
                    delay = self.get_delay(attempt)
                    logger.warning(f"Experiment {experiment_id} failed (attempt {attempt + 1}), "
                                 f"retrying in {delay:.1f}s. Error: {output[:100]}...")
                    time.sleep(delay)
                
            except Exception as e:
                error_msg = f"Unexpected exception: {str(e)}"
                failure_type = FailureType.UNKNOWN_ERROR
                self.record_failure(experiment_id, failure_type, error_msg, attempt)
                
                if not self.should_retry(experiment_id, failure_type, attempt):
                    logger.error(f"Experiment {experiment_id} failed with exception after {attempt + 1} attempts: {e}")
                    return False, error_msg
                
                if attempt < self.config.max_retries:
                    delay = self.get_delay(attempt)
                    logger.warning(f"Experiment {experiment_id} raised exception (attempt {attempt + 1}), "
                                 f"retrying in {delay:.1f}s. Exception: {str(e)}")
                    time.sleep(delay)
        
        return False, f"Failed after {self.config.max_retries + 1} attempts"
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto dei fallimenti."""
        total_experiments = len(self.failure_history)
        if total_experiments == 0:
            return {"total_experiments_with_failures": 0}
        
        total_failures = sum(len(failures) for failures in self.failure_history.values())
        
        # Conta per tipo
        failure_type_counts = {}
        for failures in self.failure_history.values():
            for failure in failures:
                failure_type = failure['failure_type']
                failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        
        # Trova esperimenti con più fallimenti
        experiments_with_most_failures = sorted(
            [(exp_id, len(failures)) for exp_id, failures in self.failure_history.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            "total_experiments_with_failures": total_experiments,
            "total_failures": total_failures,
            "average_failures_per_experiment": total_failures / total_experiments,
            "failure_type_counts": failure_type_counts,
            "experiments_with_most_failures": experiments_with_most_failures
        }
    
    def reset_failure_history(self):
        """Resetta la storia dei fallimenti."""
        self.failure_history.clear()
        logger.info("Failure history reset")

# Configurazioni predefinite
CONSERVATIVE_RETRY = RetryConfig(
    max_retries=2,
    base_delay=10.0,
    max_delay=60.0,
    exponential_backoff=True
)

AGGRESSIVE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=5.0,
    max_delay=300.0,
    exponential_backoff=True
)

MINIMAL_RETRY = RetryConfig(
    max_retries=1,
    base_delay=5.0,
    max_delay=30.0,
    exponential_backoff=False,
    retry_on_types={FailureType.TIMEOUT, FailureType.RESOURCE_ERROR}
)
