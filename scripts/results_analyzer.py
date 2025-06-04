"""
Analisi e visualizzazione dei risultati degli esperimenti di Federated Learning.

Questo modulo fornisce funzioni per analizzare i risultati raccolti dagli esperimenti
e creare visualizzazioni utili per la comprensione delle performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configurazione stile matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Analizzatore dei risultati degli esperimenti."""
    
    def __init__(self, results_path: str):
        """
        Inizializza l'analizzatore.
        
        Args:
            results_path: Percorso al file CSV con i risultati
        """
        self.results_path = Path(results_path)
        self.df = pd.read_csv(results_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepara i dati per l'analisi."""
        # Converti i tipi di dati
        self.df['round'] = self.df['round'].astype(int)
        self.df['client_id'] = self.df['client_id'].astype(int)
        self.df['run'] = self.df['run'].astype(int)
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
        
        # Rimuovi righe con valori NaN
        self.df = self.df.dropna(subset=['value'])
        
        # Crea identificatori utili
        self.df['experiment_id'] = (self.df['algorithm'] + '_' + 
                                  self.df['attack'] + '_' + 
                                  self.df['dataset'])
        
        print(f"Loaded {len(self.df)} records")
        print(f"Unique experiments: {self.df['experiment_id'].nunique()}")
        print(f"Algorithms: {sorted(self.df['algorithm'].unique())}")
        print(f"Attacks: {sorted(self.df['attack'].unique())}")
        print(f"Datasets: {sorted(self.df['dataset'].unique())}")
        print(f"Metrics: {sorted(self.df['metric'].unique())}")
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Calcola statistiche riassuntive per ogni esperimento."""
        # Filtra solo le metriche finali (ultimo round)
        final_round = self.df.groupby(['algorithm', 'attack', 'dataset', 'run'])['round'].max().reset_index()
        final_round.columns = ['algorithm', 'attack', 'dataset', 'run', 'final_round']
        
        # Merge per ottenere solo i risultati finali
        final_results = pd.merge(self.df, final_round, 
                               on=['algorithm', 'attack', 'dataset', 'run'])
        final_results = final_results[final_results['round'] == final_results['final_round']]
        
        # Calcola statistiche per ogni configurazione
        summary = final_results.groupby(['algorithm', 'attack', 'dataset', 'metric']).agg({
            'value': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        summary.columns = ['mean', 'std', 'min', 'max', 'count']
        summary = summary.reset_index()
        
        return summary
    
    def plot_convergence_curves(self, output_dir: str = "plots"):
        """Crea grafici di convergenza per ogni combinazione."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Filtra le metriche principali
        metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'f1']
        
        for dataset in self.df['dataset'].unique():
            for metric in metrics_to_plot:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'Convergence Curves - {dataset} - {metric.title()}', fontsize=16)
                
                attacks = sorted(self.df['attack'].unique())
                algorithms = sorted(self.df['algorithm'].unique())
                
                for i, attack in enumerate(attacks[:6]):  # Limite a 6 attacchi per la visualizzazione
                    row = i // 3
                    col = i % 3
                    ax = axes[row, col]
                    
                    subset = self.df[
                        (self.df['dataset'] == dataset) & 
                        (self.df['attack'] == attack) & 
                        (self.df['metric'] == metric)
                    ]
                    
                    for algorithm in algorithms:
                        alg_subset = subset[subset['algorithm'] == algorithm]
                        if len(alg_subset) > 0:
                            # Calcola media e deviazione standard per ogni round
                            round_stats = alg_subset.groupby('round')['value'].agg(['mean', 'std']).reset_index()
                            
                            ax.plot(round_stats['round'], round_stats['mean'], 
                                   label=algorithm, marker='o', markersize=4)
                            
                            # Aggiungi area di confidenza
                            if not round_stats['std'].isna().all():
                                ax.fill_between(
                                    round_stats['round'],
                                    round_stats['mean'] - round_stats['std'],
                                    round_stats['mean'] + round_stats['std'],
                                    alpha=0.2
                                )
                    
                    ax.set_title(f'Attack: {attack}')
                    ax.set_xlabel('Round')
                    ax.set_ylabel(metric.title())
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Rimuovi subplot vuoti
                for i in range(len(attacks), 6):
                    row = i // 3
                    col = i % 3
                    fig.delaxes(axes[row, col])
                
                plt.tight_layout()
                plot_filename = output_path / f"convergence_{dataset}_{metric}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved convergence plot: {plot_filename}")
    
    def plot_performance_comparison(self, output_dir: str = "plots"):
        """Crea grafici di confronto delle performance finali."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Ottieni statistiche finali
        summary = self.get_summary_statistics()
        
        # Plot per accuracy finale
        accuracy_data = summary[summary['metric'] == 'accuracy']
        
        if len(accuracy_data) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle('Final Accuracy Comparison Across Datasets', fontsize=16)
            
            for i, dataset in enumerate(sorted(accuracy_data['dataset'].unique())):
                ax = axes[i]
                dataset_data = accuracy_data[accuracy_data['dataset'] == dataset]
                
                # Crea heatmap
                pivot_data = dataset_data.pivot(index='algorithm', columns='attack', values='mean')
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                           ax=ax, cbar_kws={'label': 'Accuracy'})
                ax.set_title(f'{dataset}')
                ax.set_xlabel('Attack Type')
                ax.set_ylabel('Algorithm')
            
            plt.tight_layout()
            plot_filename = output_path / "accuracy_comparison.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved accuracy comparison: {plot_filename}")
    
    def plot_robustness_analysis(self, output_dir: str = "plots"):
        """Analizza la robustezza degli algoritmi agli attacchi."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        summary = self.get_summary_statistics()
        accuracy_data = summary[summary['metric'] == 'accuracy']
        
        if len(accuracy_data) == 0:
            print("No accuracy data found for robustness analysis")
            return
        
        # Calcola la degradazione rispetto al caso senza attacchi
        baseline_data = accuracy_data[accuracy_data['attack'] == 'none']
        
        robustness_results = []
        
        for _, row in accuracy_data.iterrows():
            if row['attack'] != 'none':
                # Trova il baseline corrispondente
                baseline = baseline_data[
                    (baseline_data['algorithm'] == row['algorithm']) &
                    (baseline_data['dataset'] == row['dataset'])
                ]
                
                if len(baseline) > 0:
                    baseline_acc = baseline['mean'].iloc[0]
                    degradation = baseline_acc - row['mean']
                    relative_degradation = degradation / baseline_acc if baseline_acc > 0 else 0
                    
                    robustness_results.append({
                        'algorithm': row['algorithm'],
                        'attack': row['attack'],
                        'dataset': row['dataset'],
                        'baseline_accuracy': baseline_acc,
                        'attack_accuracy': row['mean'],
                        'absolute_degradation': degradation,
                        'relative_degradation': relative_degradation
                    })
        
        robustness_df = pd.DataFrame(robustness_results)
        
        if len(robustness_df) > 0:
            # Plot degradazione relativa
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle('Algorithm Robustness: Relative Performance Degradation', fontsize=16)
            
            for i, dataset in enumerate(sorted(robustness_df['dataset'].unique())):
                ax = axes[i]
                dataset_data = robustness_df[robustness_df['dataset'] == dataset]
                
                pivot_data = dataset_data.pivot(index='algorithm', columns='attack', 
                                               values='relative_degradation')
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                           ax=ax, cbar_kws={'label': 'Relative Degradation'})
                ax.set_title(f'{dataset}')
                ax.set_xlabel('Attack Type')
                ax.set_ylabel('Algorithm')
            
            plt.tight_layout()
            plot_filename = output_path / "robustness_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved robustness analysis: {plot_filename}")
            
            # Salva anche i dati di robustezza
            robustness_file = output_path / "robustness_data.csv"
            robustness_df.to_csv(robustness_file, index=False)
            print(f"Saved robustness data: {robustness_file}")
    
    def plot_client_variance(self, output_dir: str = "plots"):
        """Analizza la varianza tra client."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calcola la varianza tra client per ogni esperimento e round
        client_variance = []
        
        for exp_id in self.df['experiment_id'].unique():
            exp_data = self.df[self.df['experiment_id'] == exp_id]
            
            for run in exp_data['run'].unique():
                run_data = exp_data[exp_data['run'] == run]
                
                for round_num in run_data['round'].unique():
                    round_data = run_data[run_data['round'] == round_num]
                    
                    for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
                        metric_data = round_data[round_data['metric'] == metric]
                        
                        if len(metric_data) > 1:
                            variance = metric_data['value'].var()
                            mean_val = metric_data['value'].mean()
                            
                            client_variance.append({
                                'experiment_id': exp_id,
                                'algorithm': exp_data['algorithm'].iloc[0],
                                'attack': exp_data['attack'].iloc[0],
                                'dataset': exp_data['dataset'].iloc[0],
                                'run': run,
                                'round': round_num,
                                'metric': metric,
                                'variance': variance,
                                'mean': mean_val,
                                'cv': variance / mean_val if mean_val > 0 else 0  # Coefficient of variation
                            })
        
        variance_df = pd.DataFrame(client_variance)
        
        if len(variance_df) > 0:
            # Plot evoluzione della varianza nel tempo
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Client Variance Evolution Over Rounds', fontsize=16)
            
            datasets = sorted(variance_df['dataset'].unique())
            metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1']
            
            for i, dataset in enumerate(datasets):
                for j, metric in enumerate(metrics):
                    ax = axes[j, i]
                    
                    subset = variance_df[
                        (variance_df['dataset'] == dataset) & 
                        (variance_df['metric'] == metric)
                    ]
                    
                    for attack in sorted(subset['attack'].unique()):
                        attack_data = subset[subset['attack'] == attack]
                        
                        # Media della varianza per round
                        round_variance = attack_data.groupby('round')['variance'].mean().reset_index()
                        
                        ax.plot(round_variance['round'], round_variance['variance'],
                               label=attack, marker='o', markersize=4)
                    
                    ax.set_title(f'{dataset} - {metric.title()} Variance')
                    ax.set_xlabel('Round')
                    ax.set_ylabel('Variance')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Rimuovi subplot vuoti se necessario
            for i in range(len(datasets), 3):
                for j in range(2):
                    fig.delaxes(axes[j, i])
            
            plt.tight_layout()
            plot_filename = output_path / "client_variance.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved client variance plot: {plot_filename}")
    
    def generate_summary_report(self, output_dir: str = "plots"):
        """Genera un report riassuntivo completo."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        summary = self.get_summary_statistics()
        
        # Crea report testuale
        report_lines = []
        report_lines.append("FEDERATED LEARNING EXPERIMENT SUMMARY REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total records analyzed: {len(self.df):,}")
        report_lines.append("")
        
        # Statistiche generali
        report_lines.append("EXPERIMENT OVERVIEW")
        report_lines.append("-" * 20)
        report_lines.append(f"Algorithms tested: {', '.join(sorted(self.df['algorithm'].unique()))}")
        report_lines.append(f"Attack types: {', '.join(sorted(self.df['attack'].unique()))}")
        report_lines.append(f"Datasets: {', '.join(sorted(self.df['dataset'].unique()))}")
        report_lines.append(f"Number of runs per experiment: {self.df['run'].nunique()}")
        report_lines.append("")
        
        # Best performing configurations
        accuracy_summary = summary[summary['metric'] == 'accuracy']
        if len(accuracy_summary) > 0:
            best_configs = accuracy_summary.nlargest(10, 'mean')
            
            report_lines.append("TOP 10 PERFORMING CONFIGURATIONS (by Accuracy)")
            report_lines.append("-" * 50)
            for _, config in best_configs.iterrows():
                report_lines.append(
                    f"{config['algorithm']} + {config['attack']} + {config['dataset']}: "
                    f"{config['mean']:.4f} ± {config['std']:.4f}"
                )
            report_lines.append("")
        
        # Most robust algorithms
        baseline_accuracy = accuracy_summary[accuracy_summary['attack'] == 'none']
        if len(baseline_accuracy) > 0:
            robustness_scores = []
            
            for algorithm in self.df['algorithm'].unique():
                alg_data = accuracy_summary[accuracy_summary['algorithm'] == algorithm]
                baseline = alg_data[alg_data['attack'] == 'none']['mean'].mean()
                attacked = alg_data[alg_data['attack'] != 'none']['mean'].mean()
                
                if not pd.isna(baseline) and not pd.isna(attacked):
                    robustness = attacked / baseline if baseline > 0 else 0
                    robustness_scores.append((algorithm, robustness, baseline, attacked))
            
            robustness_scores.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append("ALGORITHM ROBUSTNESS RANKING")
            report_lines.append("-" * 30)
            for i, (alg, score, baseline, attacked) in enumerate(robustness_scores, 1):
                report_lines.append(
                    f"{i}. {alg}: {score:.3f} (baseline: {baseline:.4f}, under attack: {attacked:.4f})"
                )
            report_lines.append("")
        
        # Salva il report
        report_file = output_path / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved summary report: {report_file}")
        
        # Salva anche le statistiche dettagliate
        summary_file = output_path / "detailed_statistics.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Saved detailed statistics: {summary_file}")

def main():
    """Funzione principale per l'analisi dei risultati."""
    parser = argparse.ArgumentParser(description="Analizza i risultati degli esperimenti FL")
    parser.add_argument("results_file", help="Percorso al file CSV con i risultati")
    parser.add_argument("--output-dir", default="analysis_plots", 
                       help="Directory per salvare i grafici")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Salta la generazione dei grafici")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"File non trovato: {args.results_file}")
        return
    
    # Crea l'analizzatore
    analyzer = ResultsAnalyzer(args.results_file)
    
    # Crea directory di output
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Genera il report riassuntivo
    analyzer.generate_summary_report(args.output_dir)
    
    if not args.skip_plots:
        print("Generating visualization plots...")
        
        # Genera tutti i grafici
        analyzer.plot_convergence_curves(args.output_dir)
        analyzer.plot_performance_comparison(args.output_dir)
        analyzer.plot_robustness_analysis(args.output_dir)
        analyzer.plot_client_variance(args.output_dir)
        
        print(f"All plots saved to: {output_path}")
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()
