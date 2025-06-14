#!/usr/bin/env python3
"""
Test completo per verificare tutto il flusso di parsing e salvataggio delle metriche.
Analizza:
1. Parsing delle righe di output
2. Salvataggio nel DataFrame
3. Correttezza della strategia salvata
4. Struttura del CSV finale
"""
import sys
import pandas as pd
from pathlib import Path
import re

# Add paths for reorganized imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "experiment_runners"))

from experiment_runners.enhanced_experiment_runner import EnhancedExperimentRunner, EnhancedExperimentConfig

def create_comprehensive_test_data():
    """Crea dati di test che coprono tutti i casi possibili."""
    return [
        # Round tracking
        "INFO :      [ROUND 1]",
        "INFO:flwr:[ROUND 1]",
        
        # Server metrics per fedavg
        "[Server] Round 1 aggregate fit -> accuracy=0.8500, loss=0.2100, precision=0.8400, recall=0.8600, f1=0.8500",
        "[Server] Round 1 evaluate -> accuracy=0.8200, loss=0.2500, precision=0.8100, recall=0.8300, f1=0.8200",
        
        # Client metrics per fedavg
        "[Client 0] fit | Received parameters, config: {'round': 1}",
        "[Client 0] fit complete | avg_loss=0.2100, accuracy=0.8500, precision=0.8400, recall=0.8600, f1=0.8500",
        "[Client 0] evaluate | Received parameters, config: {'round': 1}",
        "[Client 0] evaluate complete | avg_loss=0.2500, accuracy=0.8200, precision=0.8100, recall=0.8300, f1=0.8200",
        
        "[Client 1] fit | Received parameters, config: {'round': 1}",
        "[Client 1] fit complete | avg_loss=0.2200, accuracy=0.8400, precision=0.8300, recall=0.8500, f1=0.8400",
        
        # Round 2
        "INFO :      [ROUND 2]",
        "[Server] Round 2 aggregate fit -> accuracy=0.8700, loss=0.1800, precision=0.8600, recall=0.8800, f1=0.8700",
        "[Server] Round 2 evaluate -> accuracy=0.8400, loss=0.2200, precision=0.8300, recall=0.8500, f1=0.8400",
        
        "[Client 0] fit | Received parameters, config: {'round': 2}",
        "[Client 0] fit complete | avg_loss=0.1800, accuracy=0.8700, precision=0.8600, recall=0.8800, f1=0.8700",
        "[Client 1] fit | Received parameters, config: {'round': 2}",
        "[Client 1] fit complete | avg_loss=0.1900, accuracy=0.8600, precision=0.8500, recall=0.8700, f1=0.8600",
    ]

def test_strategy_parsing():
    """Test specifico per il parsing delle strategie."""
    print("=" * 80)
    print("TEST 1: PARSING DELLE STRATEGIE")
    print("=" * 80)
    
    runner = EnhancedExperimentRunner(results_dir="test_strategy_parsing", _test_mode=True)
    
    # Test con diverse strategie
    strategies_to_test = ["fedavg", "fedavgm", "fedprox", "scaffold"]
    
    for strategy in strategies_to_test:
        print(f"\n--- Testing strategy: {strategy} ---")
        
        # Crea configurazione
        config = EnhancedExperimentConfig(
            strategy=strategy,
            attack="none",
            dataset="MNIST",
            num_rounds=2,
            num_clients=2
        )
        
        # Simula l'impostazione di _actual_strategy come avviene durante l'esecuzione
        setattr(config, '_actual_strategy', strategy)
        
        run_id = 0
        
        # Test con alcune righe di esempio
        test_lines = [
            "INFO :      [ROUND 1]",
            f"[Server] Round 1 aggregate fit -> accuracy=0.8500, loss=0.2100, precision=0.8400, recall=0.8600, f1=0.8500",
            "[Client 0] fit | Received parameters, config: {'round': 1}",
            "[Client 0] fit complete | avg_loss=0.2100, accuracy=0.8500, precision=0.8400, recall=0.8600, f1=0.8500",
        ]
        
        # Processa le righe
        for line in test_lines:
            runner.parse_and_store_metrics(line, config, run_id)
        
        # Verifica che la strategia sia stata salvata correttamente
        strategy_entries = runner.results_df[runner.results_df['algorithm'] == strategy]
        
        if len(strategy_entries) > 0:
            print(f"✅ {strategy}: {len(strategy_entries)} metriche salvate correttamente")
            print(f"   Algoritmi unici nel DataFrame: {runner.results_df['algorithm'].unique()}")
        else:
            print(f"❌ {strategy}: Nessuna metrica salvata!")
    
    return runner.results_df

def test_metrics_parsing():
    """Test specifico per il parsing delle metriche."""
    print("=" * 80)
    print("TEST 2: PARSING DELLE METRICHE")
    print("=" * 80)
    
    runner = EnhancedExperimentRunner(results_dir="test_metrics_parsing", _test_mode=True)
    
    # Configurazione per fedavgm (caso problematico originale)
    config = EnhancedExperimentConfig(
        strategy="fedavgm",
        attack="none",
        dataset="MNIST",
        num_rounds=2,
        num_clients=2
    )
    setattr(config, '_actual_strategy', 'fedavgm')
    
    # Dati di test completi
    test_data = create_comprehensive_test_data()
    
    print(f"Processando {len(test_data)} righe di test...")
    
    # Processa tutte le righe
    for line in test_data:
        runner.parse_and_store_metrics(line, config, 0)
    
    df = runner.results_df
    
    print(f"\nRisultati del parsing:")
    print(f"- Righe totali nel DataFrame: {len(df)}")
    print(f"- Colonne: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"- Algoritmi unici: {df['algorithm'].unique()}")
        print(f"- Attacks unici: {df['attack'].unique()}")
        print(f"- Datasets unici: {df['dataset'].unique()}")
        print(f"- Runs unici: {df['run'].unique()}")
        print(f"- Client IDs unici: {df['client_id'].unique()}")
        print(f"- Rounds unici: {sorted(df['round'].unique())}")
        print(f"- Metriche unici: {sorted(df['metric'].unique())}")
        
        # Verifica specifiche
        fedavgm_entries = df[df['algorithm'] == 'fedavgm']
        print(f"\n- Entries per fedavgm: {len(fedavgm_entries)}")
        
        server_entries = df[df['client_id'] == 'server']
        client_entries = df[df['client_id'] != 'server']
        print(f"- Server entries: {len(server_entries)}")
        print(f"- Client entries: {len(client_entries)}")
        
        fit_entries = df[df['metric'].str.contains('fit_')]
        eval_entries = df[df['metric'].str.contains('eval_')]
        print(f"- Fit metrics: {len(fit_entries)}")
        print(f"- Eval metrics: {len(eval_entries)}")
        
        return True
    else:
        print("❌ Nessuna metrica è stata parsata!")
        return False

def test_csv_saving():
    """Test del salvataggio CSV."""
    print("=" * 80)
    print("TEST 3: SALVATAGGIO CSV")
    print("=" * 80)
    
    runner = EnhancedExperimentRunner(results_dir="test_csv_saving", _test_mode=True)
    
    # Aggiungi alcuni dati di test
    config = EnhancedExperimentConfig(
        strategy="fedavgm",
        attack="noise",
        dataset="FMNIST",
        attack_params={"noise_std": 0.1},
        num_rounds=2,
        num_clients=2
    )
    setattr(config, '_actual_strategy', 'fedavgm')
    
    test_data = create_comprehensive_test_data()
    
    for line in test_data:
        runner.parse_and_store_metrics(line, config, 0)
    
    print(f"DataFrame shape prima del salvataggio: {runner.results_df.shape}")
    
    # Salva i risultati
    runner.save_results(intermediate=False)
    
    # Verifica che i file siano stati creati
    results_dir = Path("test_csv_saving")
    csv_files = list(results_dir.glob("final_results_*.csv"))
    json_files = list(results_dir.glob("final_results_*.json"))
    
    print(f"File CSV creati: {len(csv_files)}")
    print(f"File JSON creati: {len(json_files)}")
    
    if csv_files:
        # Leggi il CSV e verifica il contenuto
        csv_file = csv_files[0]
        print(f"Leggendo: {csv_file}")
        
        loaded_df = pd.read_csv(csv_file)
        print(f"CSV caricato: {loaded_df.shape}")
        print(f"Colonne nel CSV: {list(loaded_df.columns)}")
        
        if len(loaded_df) > 0:
            print(f"Sample del CSV:")
            print(loaded_df.head().to_string())
            
            # Verifica specifica per fedavgm
            fedavgm_in_csv = loaded_df[loaded_df['algorithm'] == 'fedavgm']
            print(f"\nRows con fedavgm nel CSV: {len(fedavgm_in_csv)}")
            
            if len(fedavgm_in_csv) > 0:
                print("✅ fedavgm salvato correttamente nel CSV!")
                return True
            else:
                print("❌ fedavgm NON trovato nel CSV!")
                return False
        else:
            print("❌ CSV vuoto!")
            return False
    else:
        print("❌ Nessun file CSV creato!")
        return False

def test_regex_patterns():
    """Test dei pattern regex utilizzati."""
    print("=" * 80)
    print("TEST 4: PATTERN REGEX")
    print("=" * 80)
    
    # Test delle righe reali contro i pattern regex
    test_lines = [
        "INFO :      [ROUND 1]",
        "[Server] Round 1 aggregate fit -> accuracy=0.2100, loss=2.1999, precision=0.1897, recall=0.2037, f1=0.1860",
        "[Server] Round 1 evaluate -> accuracy=0.4400, loss=1.7303, precision=0.4654, recall=0.4481, f1=0.3904",
        "[Client 0] fit | Received parameters, config: {'round': 1}",
        "[Client 0] fit complete | avg_loss=2.1999, accuracy=0.2100, precision=0.1897, recall=0.2037, f1=0.1860",
        "[Client 0] evaluate | Received parameters, config: {'round': 1}",
        "[Client 0] evaluate complete | avg_loss=1.7303, accuracy=0.4400, precision=0.4654, recall=0.4481, f1=0.3904",
    ]
    
    patterns = {
        "round": r"\[ROUND (\d+)\]",
        "server_fit": r"\[Server\] Round (\d+) aggregate fit -> (.+)",
        "server_eval": r"\[Server\] Round (\d+) evaluate -> (.+)",
        "client_fit": r"\[Client (\d+)\] fit complete \| (.+)",
        "client_eval": r"\[Client (\d+)\] evaluate complete \| (.+)",
        "client_config": r"\[Client (\d+)\] (?:fit|evaluate) \| Received parameters, config: \{'round': (\d+)\}",
        "metrics": r"(\w+)=([\d\.]+)"
    }
    
    for line in test_lines:
        print(f"\nTestando: {line}")
        
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                print(f"  ✅ {pattern_name}: {match.groups()}")
                
                # Test specifico per metrics extraction
                if pattern_name in ["server_fit", "server_eval", "client_fit", "client_eval"]:
                    if len(match.groups()) > 1:
                        metrics_str = match.groups()[-1]
                        metric_matches = re.findall(patterns["metrics"], metrics_str)
                        print(f"     Metriche estratte: {metric_matches}")

def main():
    """Esegui tutti i test."""
    print("🧪 TEST COMPLETO DEL FLUSSO DI PARSING E SALVATAGGIO")
    print("=" * 80)
    
    try:
        # Test 1: Pattern regex
        test_regex_patterns()
        
        # Test 2: Parsing strategie
        strategies_df = test_strategy_parsing()
        
        # Test 3: Parsing metriche
        metrics_success = test_metrics_parsing()
        
        # Test 4: Salvataggio CSV
        csv_success = test_csv_saving()
        
        print("\n" + "=" * 80)
        print("RISULTATI FINALI")
        print("=" * 80)
        
        if metrics_success and csv_success:
            print("🎉 TUTTI I TEST SUPERATI!")
            print("✅ Parsing delle metriche funziona correttamente")
            print("✅ Salvataggio CSV funziona correttamente")
            print("✅ fedavgm viene salvato correttamente")
            return True
        else:
            print("💥 ALCUNI TEST FALLITI!")
            if not metrics_success:
                print("❌ Problema nel parsing delle metriche")
            if not csv_success:
                print("❌ Problema nel salvataggio CSV")
            return False
            
    except Exception as e:
        print(f"💥 ERRORE DURANTE I TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
