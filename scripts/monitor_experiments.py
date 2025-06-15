#!/usr/bin/env python3
"""
Script per monitorare il progresso degli esperimenti in corso.

Fornisce informazioni in tempo reale sui checkpoint, statistiche e tempo stimato.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utilities.checkpoint_manager import CheckpointManager
from configuration.config_manager import get_config_manager

def format_duration(seconds):
    """Formatta una durata in secondi in formato leggibile."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def estimate_completion_time(checkpoint_mgr: CheckpointManager):
    """Stima il tempo di completamento basandosi sui progressi attuali."""
    progress = checkpoint_mgr.get_progress_summary()
    
    if progress['completed_runs'] == 0:
        return None, None
    
    # Calcola tempo medio per run osservando i timestamp
    timestamps = []
    for experiment_id, status in checkpoint_mgr.experiment_status.items():
        if status.completed_runs:
            try:
                last_updated = datetime.fromisoformat(status.last_updated)
                timestamps.append(last_updated)
            except:
                continue
    
    if len(timestamps) < 2:
        return None, None
    
    # Stima basandosi sui timestamp disponibili
    timestamps.sort()
    time_span = (timestamps[-1] - timestamps[0]).total_seconds()
    runs_in_span = progress['completed_runs']
    
    if runs_in_span > 0 and time_span > 0:
        avg_time_per_run = time_span / runs_in_span
        remaining_runs = progress['remaining_runs']
        estimated_seconds = remaining_runs * avg_time_per_run
        
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        return avg_time_per_run, estimated_completion
    
    return None, None

def show_progress_summary(checkpoint_mgr: CheckpointManager):
    """Mostra riassunto del progresso."""
    progress = checkpoint_mgr.get_progress_summary()
    
    print("="*60)
    print("EXPERIMENT PROGRESS SUMMARY")
    print("="*60)
    
    # Se non ci sono esperimenti, mostra un messaggio appropriato
    if progress['total_experiments'] == 0:
        print("No experiments found in checkpoint data.")
        print()
        print("This could mean:")
        print("1. Experiments haven't been started yet")
        print("2. Checkpoint files are corrupted or empty")
        print("3. Wrong checkpoint directory specified")
        print("="*60)
        return
    
    print(f"Total Experiments: {progress['total_experiments']}")
    print(f"Completed Experiments: {progress['completed_experiments']}")
    print(f"Remaining Experiments: {progress['remaining_experiments']}")
    print()
    
    print(f"Total Runs: {progress['total_runs']}")
    print(f"Completed Runs: {progress['completed_runs']}")
    print(f"Failed Runs: {progress['failed_runs']}")
    print(f"Remaining Runs: {progress['remaining_runs']}")
    print()
    
    if progress['total_runs'] > 0:
        completion_percent = progress['overall_progress']
        print(f"Overall Progress: {completion_percent:.1f}%")
        
        # Barra di progresso
        bar_length = 40
        filled_length = int(bar_length * completion_percent / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"Progress Bar: [{bar}] {completion_percent:.1f}%")
        
        if progress['completed_runs'] > 0:
            success_rate = (progress['completed_runs'] / (progress['completed_runs'] + progress['failed_runs'])) * 100
            print(f"Success Rate: {success_rate:.1f}%")
    
    print()
    
    # Stima tempo di completamento
    avg_time, completion_time = estimate_completion_time(checkpoint_mgr)
    if avg_time and completion_time:
        print(f"Average time per run: {format_duration(avg_time)}")
        print(f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        time_remaining = (completion_time - datetime.now()).total_seconds()
        if time_remaining > 0:
            print(f"Time remaining: {format_duration(time_remaining)}")
    
    print("="*60)

def show_detailed_status(checkpoint_mgr: CheckpointManager):
    """Mostra stato dettagliato degli esperimenti."""
    print("\nDETAILED EXPERIMENT STATUS")
    print("-"*60)
    
    if not checkpoint_mgr.experiment_status:
        print("No experiment status data available.")
        return
    
    experiments_by_status = {
        'completed': [],
        'in_progress': [],
        'failed': [],
        'not_started': []
    }
    
    for experiment_id, status in checkpoint_mgr.experiment_status.items():
        if status.is_complete():
            if len(status.failed_runs) == 0:
                experiments_by_status['completed'].append((experiment_id, status))
            else:
                experiments_by_status['failed'].append((experiment_id, status))
        elif len(status.completed_runs) > 0:
            experiments_by_status['in_progress'].append((experiment_id, status))
        else:
            experiments_by_status['not_started'].append((experiment_id, status))
    
    for category, experiments in experiments_by_status.items():
        if experiments:
            print(f"\n{category.upper()} ({len(experiments)} experiments):")
            for experiment_id, status in experiments[:10]:  # Mostra solo i primi 10
                completed = len(status.completed_runs)
                failed = len(status.failed_runs)
                total = status.total_runs
                print(f"  {experiment_id}: {completed}/{total} completed, {failed} failed")
            
            if len(experiments) > 10:
                print(f"  ... and {len(experiments) - 10} more")

def show_failure_analysis(checkpoint_mgr: CheckpointManager):
    """Analizza i fallimenti negli esperimenti."""
    print("\nFAILURE ANALYSIS")
    print("-"*60)
    
    if not checkpoint_mgr.experiment_status:
        print("No experiment status data available.")
        return
    
    total_failures = sum(len(status.failed_runs) for status in checkpoint_mgr.experiment_status.values())
    
    if total_failures == 0:
        print("No failures recorded.")
        return
    
    print(f"Total failed runs: {total_failures}")
    
    # Esperimenti con più fallimenti
    experiments_with_failures = [
        (exp_id, len(status.failed_runs))
        for exp_id, status in checkpoint_mgr.experiment_status.items()
        if status.failed_runs
    ]
    
    if experiments_with_failures:
        experiments_with_failures.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nExperiments with most failures:")
        for exp_id, failure_count in experiments_with_failures[:5]:
            print(f"  {exp_id}: {failure_count} failures")

def show_backup_info(checkpoint_mgr: CheckpointManager):
    """Mostra informazioni sui backup."""
    print("\nBACKUP INFORMATION")
    print("-"*60)
    
    backup_dir = checkpoint_mgr.results_backup_dir
    
    if not backup_dir.exists():
        print("No backup directory found.")
        return
    
    backup_files = list(backup_dir.glob("results_backup_*.csv"))
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not backup_files:
        print("No backup files found.")
        return
    
    print(f"Found {len(backup_files)} backup files:")
    
    for i, backup_file in enumerate(backup_files[:5]):  # Mostra solo i 5 più recenti
        mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
        size_mb = backup_file.stat().st_size / (1024 * 1024)
        print(f"  {backup_file.name}: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({size_mb:.1f} MB)")
    
    if len(backup_files) > 5:
        print(f"  ... and {len(backup_files) - 5} older backups")

def watch_mode(checkpoint_mgr: CheckpointManager, refresh_interval: int = 10):
    """Modalità watch per monitoraggio continuo."""
    print(f"Starting watch mode (refresh every {refresh_interval}s, press Ctrl+C to exit)...")
    print()
    
    try:
        while True:
            # Cancella schermo (compatible con Windows e Unix)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Ricarica stato dal disco
            checkpoint_mgr.load_state()
            
            show_progress_summary(checkpoint_mgr)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nWatch mode stopped.")

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Monitor experiment progress")
    parser.add_argument("--results-dir", type=str, default="experiment_runners/extensive_results",
                       help="Directory containing experiment results")
    parser.add_argument("--watch", action="store_true",
                       help="Watch mode for continuous monitoring")
    parser.add_argument("--refresh-interval", type=int, default=10,
                       help="Refresh interval for watch mode (seconds)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed experiment status")
    parser.add_argument("--failures", action="store_true",
                       help="Show failure analysis")
    parser.add_argument("--backups", action="store_true",
                       help="Show backup information")
    
    args = parser.parse_args()
    
    # Inizializza checkpoint manager
    results_dir = Path(args.results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"No checkpoint directory found at {checkpoint_dir}")
        print("Make sure you're running this from the correct directory and that")
        print("extensive experiments have been started.")
        sys.exit(1)    
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Verifica se ci sono file di checkpoint
    status_file = checkpoint_dir / "experiment_status.json"
    config_file = checkpoint_dir / "configurations.json"
    
    print(f"Checking checkpoint directory: {checkpoint_dir}")
    print(f"Status file exists: {status_file.exists()}")
    print(f"Config file exists: {config_file.exists()}")
    
    if status_file.exists() or config_file.exists():
        # Forza ricaricamento dello stato
        checkpoint_mgr.load_state()
        
        print(f"Loaded {len(checkpoint_mgr.experiment_status)} experiments")
        print(f"Loaded {len(checkpoint_mgr.configurations)} configurations")
        
        # Se non ci sono dati dopo il caricamento, mostra informazioni di debug
        if not checkpoint_mgr.experiment_status and not checkpoint_mgr.configurations:
            print("\nDEBUG: Checkpoint files exist but contain no data")
            
            # Controlla contenuto dei file
            if status_file.exists():
                try:
                    with open(status_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"Status file content preview: {content[:200]}...")
                        else:
                            print("Status file is empty")
                except Exception as e:
                    print(f"Error reading status file: {e}")
            
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"Config file content preview: {content[:200]}...")
                        else:
                            print("Config file is empty")
                except Exception as e:
                    print(f"Error reading config file: {e}")
        
        # Mostra i dati anche se sono vuoti (per debugging)
        if args.watch:
            watch_mode(checkpoint_mgr, args.refresh_interval)
        else:
            show_progress_summary(checkpoint_mgr)
            if args.detailed:
                show_detailed_status(checkpoint_mgr)
            
            if args.failures:
                show_failure_analysis(checkpoint_mgr)
            
            if args.backups:
                show_backup_info(checkpoint_mgr)
    else:
        print("No checkpoint files found.")
        print(f"Expected files:")
        print(f"  - {status_file}")
        print(f"  - {config_file}")
        
        # Elenca i file presenti nella directory checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*"))
        if checkpoint_files:
            print(f"\nFiles found in checkpoint directory:")
            for file_path in sorted(checkpoint_files):
                file_type = "DIR" if file_path.is_dir() else "FILE"
                size = f"({file_path.stat().st_size} bytes)" if file_path.is_file() else ""
                print(f"  [{file_type}] {file_path.name} {size}")
        else:
            print("\nCheckpoint directory is empty.")
        
        print("Make sure extensive experiments have been started.")

if __name__ == "__main__":
    main()
