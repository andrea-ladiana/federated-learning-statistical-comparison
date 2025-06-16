# Federated Learning Framework with Flower

A comprehensive federated learning framework built on top of the Flower library, implementing multiple aggregation strategies, neural network models, datasets, and attack scenarios for research and educational purposes.

## 🌟 Key Highlights

- **10 Federated Learning Strategies** including robust Byzantine-fault tolerant algorithms
- **Enhanced Experiment Runner** with parallel execution, checkpointing, and monitoring
- **6 Attack Types** for security research and defense evaluation
- **3 Popular Datasets** (MNIST, Fashion-MNIST, CIFAR-10) with IID and Non-IID partitioning support
- **6 Neural Network Models** optimized for different computational constraints and datasets
- **Comprehensive Baseline Integration** from the Flower ecosystem
- **Advanced Utilities** including checkpoint management, retry mechanisms, and resource monitoring
- **Educational Documentation** with scientific references and compliance reports



## Table of Contents

- [Recent Reorganization & Architecture](#recent-reorganization--architecture)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Runners](#experiment-runners)
- [Enhanced Experiment Runner](#enhanced-experiment-runner)
- [Configuration Management](#configuration-management)
- [Command-Line Usage](#command-line-usage)
- [Aggregation Strategies](#aggregation-strategies)
- [Neural Network Models](#neural-network-models)
- [Datasets](#datasets)
- [Attack Implementations](#attack-implementations)
- [Utilities & Tools](#utilities--tools)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Baselines and Benchmarks](#baselines-and-benchmarks)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [Documentation](#documentation)

## Overview

This federated learning framework provides a complete research and educational environment for experimenting with federated learning algorithms. It includes:

- **10 federated learning strategies** (custom implementations with scientific validation)
- **Enhanced experiment runner** with parallel execution, checkpointing, and resource monitoring
- **6 neural network models** optimized for different computational constraints and datasets
- **3 popular datasets** (MNIST, Fashion-MNIST, CIFAR-10) with flexible partitioning
- **6 attack types** for comprehensive security research and defense evaluation
- **Comprehensive baseline integrations** from the Flower ecosystem
- **Advanced utilities** for checkpoint management, retry handling, and experiment monitoring
- **Educational documentation** with scientific references, compliance reports, and implementation details
- **Configuration management** with YAML-based settings and environment validation
- **Results analysis tools** for statistical evaluation and visualization

## Features

### 🚀 Core Capabilities
- ✅ **Multi-strategy federated learning simulation** with 10 scientifically validated algorithms
- ✅ **Robust aggregation algorithms** including Byzantine-fault tolerant methods (Krum, Bulyan, TrimmedMean)
- ✅ **Attack simulation and defense evaluation** with 6 different attack types
- ✅ **Non-IID data distribution support** with configurable heterogeneity levels
- ✅ **Comprehensive model zoo** from lightweight (702 parameters) to complex architectures
- ✅ **Educational documentation** with scientific papers and implementation guides

### 🔬 Research Features
- ✅ **Enhanced experiment runner** with parallel execution and resource monitoring
- ✅ **Byzantine fault tolerance evaluation** with configurable malicious clients
- ✅ **Adversarial attack simulations** with parameter sweeps and intensity control
- ✅ **Statistical heterogeneity handling** with advanced partitioning strategies
- ✅ **Communication efficiency optimization** through various aggregation methods
- ✅ **Convergence analysis tools** with comprehensive metrics collection
- ✅ **Results analysis and visualization** with automated report generation

### 📊 Experiment Management
- ✅ **Automated experiment execution** with YAML configuration management
- ✅ **Multi-run statistical analysis** for robust results
- ✅ **Advanced checkpointing system** with automatic recovery
- ✅ **Resource monitoring** with CPU, memory, and process tracking
- ✅ **Intelligent retry mechanisms** with exponential backoff
- ✅ **Parallel execution support** with configurable worker limits
- ✅ **Comprehensive logging** with detailed progress tracking
- ✅ **CSV and JSON export** for further analysis
- ✅ **Error handling and recovery** for long-running experiments

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd federated-learning-statistical-comparison
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Alternative: Use conda environment:**
```bash
conda env create -f configuration/environment.yml
conda activate federated-learning
```

5. **Setup and verification:**
```bash
# Quick setup
python setup.py

# Detailed setup with testing (if available locally)
# python maintenance/setup_and_test.py

# Install Flower (if available locally)  
# python maintenance/install_flower.py
```

6. **Verify installation:**
```bash
# Test models work correctly
python scripts/verify_models.py

# Run a quick test
python experiment_runners/enhanced_experiment_runner.py --help
```

## Quick Start

> 🆕 **Enhanced Experiment Runner**: Use the enhanced experiment runner for parallel execution, checkpointing, and advanced monitoring.

### Basic Federated Learning

Run a simple federated learning simulation with FedAvg strategy:

```bash
python core/server.py --strategy fedavg --model CNNNet --dataset mnist --num_clients 10 --num_rounds 10
```

### Enhanced Experiment Runner (Recommended)

Use the enhanced experiment runner for advanced features:

```bash
# Run enhanced experiments with default configuration
python experiment_runners/enhanced_experiment_runner.py

# Run with custom configuration
python experiment_runners/enhanced_experiment_runner.py --config configuration/enhanced_config.yaml

# Run in parallel mode with specific number of workers
python experiment_runners/enhanced_experiment_runner.py --parallel --max-workers 4

# Run extensive experiments with full configuration matrix
python experiment_runners/run_extensive_experiments.py

# Test mode with minimal configurations
python experiment_runners/enhanced_experiment_runner.py --test-mode
```

### With Attack Simulation

Run federated learning with adversarial attacks:

```bash
python experiment_runners/run_with_attacks.py --strategy fedavg --dataset cifar10 \
    --attack labelflip --labelflip-fraction 0.2 --num-clients 10
# Optional: reduce wait time for server start (default 5s)
python experiment_runners/run_with_attacks.py --strategy fedavg --dataset cifar10 \
    --attack labelflip --labelflip-fraction 0.2 --num-clients 10 \
    --server-start-timeout 2
```

### Setup and Verification

```bash
# Quick setup and environment check
python setup.py

# Verify models work after reorganization
python scripts/verify_models.py
```

### Monitoring and Analysis

```bash
# Monitor running experiments in real-time
python scripts/monitor_experiments.py --watch

# Analyze results with visualization
python scripts/results_analyzer.py --results-file enhanced_results/experiment_results.csv
```

## 🧪 Experiment Runners

> 📁 **Reorganized**: Experiment runners are now organized in the `experiment_runners/` directory with different capabilities.

The framework includes multiple experiment runners for different research needs:

### Available Runners

1. **`basic_experiment_runner.py`** - Basic systematic experiment execution
2. **`stable_experiment_runner.py`** - Stable version with enhanced error handling
3. **`enhanced_experiment_runner.py`** - Advanced runner with parallel execution, checkpoints, and monitoring
4. **`run_extensive_experiments.py`** - Comprehensive experiment suite
5. **`run_with_attacks.py`** - Specialized attack simulation orchestrator

### Key Features

- **Automated experiment execution** across multiple strategies, attacks, and datasets
- **Multi-run statistical analysis** for robust experimental results
- **Comprehensive logging** with real-time progress tracking
- **Intermediate result saving** to prevent data loss during long experiments
- **Error handling and recovery** for robust execution
- **CSV and JSON export** for analysis in external tools
- **🆕 Parallel execution** and checkpoint/resume functionality (enhanced runner)

### Basic Usage

#### Using the Unified Interface (Recommended)
```bash
# Run enhanced experiments with automatic configuration
python run_experiments.py --runner enhanced --num-runs 10

# Run in test mode with reduced configurations
python run_experiments.py --runner enhanced --test-mode --num-runs 3

# Run with parallel execution
python run_experiments.py --runner enhanced --num-runs 10 --parallel
```

#### Direct Runner Usage
```bash
# Basic experiment runner
python experiment_runners/basic_experiment_runner.py --num-runs 10

# Enhanced runner with advanced features
python experiment_runners/enhanced_experiment_runner.py --num-runs 10 --parallel

# Extensive experiments
python experiment_runners/run_extensive_experiments.py --num-runs 5
```

#### Experiment Configuration

The experiment runner automatically generates configurations for:

**Strategies (19 total):**
- Standard FL: `fedavg`, `fedavgm`, `fedprox`, `fednova`, `scaffold`, `fedadam`
- Byzantine-robust: `krum`, `trimmedmean`, `bulyan`
- Flower baselines: `dasha`, `depthfl`, `heterofl`, `fedmeta`, `fedper`, `fjord`, `flanders`, `fedopt`

**Attacks (7 configurations):**
- `none`: No attack (baseline)
- `noise`: Gaussian noise injection (σ=0.1, 30% clients)
- `missed`: Missed participation (30% probability)
- `failure`: Client failures (20% probability)
- `asymmetry`: Data asymmetry (factor=0.5-3.0)
- `labelflip`: Label flipping (20% clients, 80% flip probability)
- `gradflip`: Gradient flipping (20% clients, intensity=1.0)

**Datasets:** `MNIST`, `FMNIST`, `CIFAR10`

## 🚀 Enhanced Experiment Runner

For large-scale research studies that may run for days, we provide an **Enhanced Experiment Runner** (`enhanced_experiment_runner.py`) with advanced checkpoint/resume functionality, parallel execution, and comprehensive monitoring.

### 🔧 Key Enhanced Features

- **🔄 Checkpoint/Resume System**: Automatic state persistence with recovery from interruptions
- **⚡ Parallel Execution**: Thread-based parallel processing with configurable worker limits
- **📊 Real-time Monitoring**: CPU/memory/resource tracking during experiment execution
- **🛡️ Robust Error Handling**: Intelligent retry system with exponential backoff
- **📋 Centralized Configuration**: YAML-based configuration management with validation
- **🔍 Advanced Validation**: Parameter consistency checks and environment validation
- **📈 Comprehensive Metrics**: Detailed performance and statistical analysis
- **🔄 Port Management**: Automatic port allocation for parallel experiments
- **🎯 Attack Integration**: Seamless integration with attack simulations

### 📚 Enhanced System Usage

#### Quick Start with Enhanced Runner

```bash
# Run experiments with enhanced features
python experiment_runners/enhanced_experiment_runner.py

# Run with custom configuration
python experiment_runners/enhanced_experiment_runner.py --config configuration/enhanced_config.yaml

# Run extensive experiments with parallel execution
python experiment_runners/run_extensive_experiments.py --num-runs 10

# Monitor running experiments in real-time
python scripts/monitor_experiments.py --watch --refresh-interval 30
```

#### Configuration Management

The enhanced system uses YAML-based configuration (`configuration/enhanced_config.yaml`):

```yaml
system:
  max_retries: 2
  retry_delay: 5
  process_timeout: 120
  port: 8080
  log_level: "INFO"
  max_parallel_experiments: 1
  resource_monitoring: true
  checkpoint_interval: 10

defaults:
  num_rounds: 10
  num_clients: 10
  learning_rate: 0.01
  batch_size: 32
```

#### Advanced Features

```python
# Use the enhanced experiment runner programmatically
from experiment_runners.enhanced_experiment_runner import EnhancedExperimentRunner

# Initialize with configuration
runner = EnhancedExperimentRunner(config_path="configuration/enhanced_config.yaml")

# Run experiments with monitoring
results = runner.run_experiments()
```
config_manager = EnhancedConfigManager("custom_config.yaml")
runner = EnhancedExperimentRunner(config_manager=config_manager)

# Run experiments with parallel execution
results = runner.run_experiments_parallel(
    configs=experiment_configs,
    num_runs=10,
    max_workers=4,
    checkpoint_enabled=True
)
```

### 🔄 Checkpoint System

The enhanced checkpoint system provides automatic recovery for long-running experiments:

#### Features
- **Automatic State Saving**: Progress saved every 60 seconds (configurable)
- **Granular Recovery**: Resume from exact point of interruption
- **Failure Tracking**: Comprehensive error logging and retry management
- **Configuration Validation**: Ensures consistency across resume sessions
- **Backup Management**: Automatic backup rotation and cleanup

#### Usage
```bash
# Start extensive experiments (automatically creates checkpoints)
python run_extensive_experiments.py --num-runs 50

# If interrupted, resume with:
python run_extensive_experiments.py --resume

# Check progress without resuming:
python monitor_experiments.py --detailed --failures
```

### 📊 Real-time Monitoring

Monitor experiment progress with the built-in monitoring system:

```bash
# Watch mode with auto-refresh
python monitor_experiments.py --watch --refresh-interval 10

# Detailed status report
python monitor_experiments.py --detailed

# Failure analysis
python monitor_experiments.py --failures

# Backup information
python monitor_experiments.py --backups
```

#### Monitoring Output Example
```
EXPERIMENT PROGRESS SUMMARY
============================================================
Total Experiments: 399
Completed Experiments: 127
Remaining Experiments: 272

Total Runs: 3990
Completed Runs: 1847
Failed Runs: 23
Remaining Runs: 2120

Overall Progress: 46.3%
Progress Bar: [██████████████████░░░░░░░░░░░░░░░░░░░░] 46.3%
Success Rate: 98.8%

Average time per run: 127.4s
Estimated completion: 2025-06-05 14:32:18
Time remaining: 7h 23m 45s
```

### 🧪 Testing Framework

Comprehensive testing suite for the enhanced system:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --test-type unit
python tests/run_tests.py --test-type integration
python tests/run_tests.py --test-type performance

# Run with coverage
python tests/run_tests.py --coverage
```

#### Test Categories
- **Unit Tests**: Component-level testing for all major classes
- **Integration Tests**: Checkpoint system and recovery testing
- **Performance Tests**: Large-scale experiment simulation
- **Stress Tests**: Resource management under high load

### 🔧 Error Handling & Recovery

The enhanced system includes sophisticated error handling:

#### Retry System
- **Intelligent Classification**: Automatic failure type detection
- **Exponential Backoff**: Smart retry timing with configurable delays
- **Selective Retry**: Only retry recoverable failures
- **Failure Tracking**: Comprehensive failure history and analysis

#### Error Types
- `TIMEOUT`: Process timeout (retryable)
- `RESOURCE_ERROR`: Memory/CPU/Port issues (retryable)
- `NETWORK_ERROR`: Connection problems (retryable)
- `PROCESS_ERROR`: Application-level errors (non-retryable)
- `PERMISSION_ERROR`: Access denied (non-retryable)

### 📈 Performance Optimization

#### Resource Management
- **CPU Monitoring**: Real-time CPU usage tracking with thresholds
- **Memory Tracking**: Automatic memory monitoring and alerts
- **Port Management**: Dynamic port allocation for parallel experiments
- **Process Cleanup**: Automatic cleanup of orphaned processes

#### Parallel Execution
- **Thread Pool**: Efficient thread-based parallel processing
- **Resource Allocation**: Smart resource distribution across experiments
- **Load Balancing**: Automatic load balancing for optimal performance
- **Scalability**: Support for large-scale multi-day experiments

### Advanced Configuration
```python
from experiment_runner import ExperimentRunner, ExperimentConfig

# Create custom configurations
configs = [
    ExperimentConfig(
        strategy="fedavg",
        attack="labelflip",
        dataset="MNIST",
        attack_params={"labelflip_fraction": 0.3, "flip_prob": 0.9},
        strategy_params={},
        num_rounds=10,
        num_clients=10
    ),
    ExperimentConfig(
        strategy="fedprox",
        attack="noise",
        dataset="CIFAR10",
        attack_params={"noise_std": 0.2, "noise_fraction": 0.4},
        strategy_params={"proximal_mu": 0.01},
        num_rounds=15,
        num_clients=20
    )
]

# Run experiments
runner = ExperimentRunner(results_dir="custom_results")
results_df = runner.run_experiments(configs, num_runs=5)
```

#### Results Analysis

The experiment runner generates comprehensive results in long-form pandas DataFrame:

```python
import pandas as pd

# Load results
results = pd.read_csv("experiment_results/final_results_20250529_203531.csv")

# Analyze results
print(f"Total experiments: {len(results['run'].unique())}")
print(f"Strategies tested: {results['algorithm'].nunique()}")
print(f"Metrics collected: {results['metric'].nunique()}")

# Group by strategy and attack
summary = results.groupby(['algorithm', 'attack', 'metric'])['value'].agg(['mean', 'std', 'count'])
print(summary)
```

### Output Structure

The experiment runner generates:

```
experiment_results/
├── final_results_YYYYMMDD_HHMMSS.csv      # Complete results in long format
├── final_results_YYYYMMDD_HHMMSS.json     # Backup in JSON format
├── intermediate_results_*.csv              # Saved every 5 experiments
└── experiment_runner.log                   # Detailed execution logs
```

**Result DataFrame columns:**
- `algorithm`: Federated learning strategy used
- `attack`: Attack type applied (includes parameters for identification)
- `dataset`: Dataset used for training
- `run`: Run number (for statistical analysis)
- `client_id`: Client identifier (-1 for server metrics)
- `round`: Federated learning round
- `metric`: Type of metric (loss, accuracy, precision, recall, f1, eval_loss, eval_accuracy, server_loss, server_accuracy)
- `value`: Metric value

### Statistical Analysis

#### Performance Metrics

- **Final Accuracy**: Model accuracy after training completion
- **Precision**: Macro-averaged precision across classes
- **Recall**: Macro-averaged recall across classes
- **F1 Score**: Harmonic mean of precision and recall
- **Convergence Rate**: Rounds needed to reach target accuracy
- **Stability**: Variance across multiple runs
- **Robustness**: Performance degradation under attacks

#### Attack Effectiveness Metrics

- **Attack Success Rate**: Percentage of successful attacks
- **Performance Degradation**: Changes in accuracy, loss, precision, recall, and F1 score due to attacks
- **Recovery Time**: Rounds needed to recover from attacks
- **Defense Effectiveness**: Robustness of Byzantine-fault tolerant strategies

#### Example Analysis Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load experimental results
df = pd.read_csv("experiment_results/final_results_20250529_203531.csv")

# Filter for final round server accuracy
final_accuracy = df[
    (df['metric'] == 'server_accuracy') & 
    (df['round'] == df['round'].max()) &
    (df['client_id'] == -1)
]

# Group by strategy and attack
grouped = final_accuracy.groupby(['algorithm', 'attack'])['value'].agg(['mean', 'std', 'count'])

# Statistical significance testing
from scipy import stats
baseline = final_accuracy[final_accuracy['attack'] == 'none']
attacked = final_accuracy[final_accuracy['attack'] != 'none']

t_stat, p_value = stats.ttest_ind(baseline['value'], attacked['value'])
print(f"Attack impact significance: p-value = {p_value:.4f}")

# Visualization
plt.figure(figsize=(12, 8))
sns.boxplot(data=final_accuracy, x='algorithm', y='value', hue='attack')
plt.xticks(rotation=45)
plt.title('Strategy Performance Under Different Attacks')
plt.ylabel('Final Accuracy')
plt.tight_layout()
plt.savefig('strategy_attack_comparison.png', dpi=300)
```

### Reproducibility

All experimental results include:

- **Random seed tracking** for reproducible experiments
- **Parameter logging** with complete configuration details
- **Environment information** including library versions
- **Execution metadata** with timestamps and system information

### Research Applications

This framework has been used for:

- **Algorithm comparison studies** evaluating 19 different FL strategies
- **Attack simulation research** testing defense mechanisms
- **Non-IID robustness analysis** across heterogeneous data distributions
- **Communication efficiency studies** comparing aggregation methods
- **Educational demonstrations** for FL courses and workshops

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Run tests:
   ```bash
   python -m pytest tests/
   ```

### Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Document all functions and classes
- Add unit tests for new features

### Adding New Components

1. **New Strategy:** Add to `strategy/` and register in `strategies.py`
2. **New Model:** Add to `models/` and update `__init__.py`
3. **New Attack:** Add to `attacks/` and register in `attack_config.py`
4. **New Dataset:** Update data loading logic in `client.py`

## Documentation

### Scientific References

This framework implements algorithms from established research papers. For detailed information about the theoretical foundations and original implementations, please refer to the scientific literature cited in the code documentation.


### Examples and Tutorials

See the `examples/` directory for:
- Basic federated learning tutorial
- Advanced strategy comparison
- Attack simulation examples
- Hyperparameter tuning guide
- Performance optimization tips

## Performance Optimization

### Training Speed
- Use GPU acceleration when available
- Optimize batch sizes for your hardware
- Consider model parallelism for large models
- Use efficient data loading with multiple workers

### Memory Usage
- Monitor memory consumption with lightweight models
- Use gradient accumulation for large batches
- Consider federated learning with compressed communication

### Communication Efficiency
- Use strategies like FedNova for fewer rounds
- Implement gradient compression
- Optimize client sampling strategies

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size or use CPU
   python server.py --strategy fedavg --batch_size 16 --device cpu
   ```

2. **Port Already in Use:**
   ```bash
   # Kill existing Flower processes
   python experiment_runner.py  # Automatically handles this
   # Or manually:
   pkill -f "python.*server.py"
   ```

3. **Client Connection Issues:**
   ```bash
   # Check network configuration and firewall
   python server.py --verbose --host 0.0.0.0
   ```

4. **Convergence Problems:**
   ```bash
   # Adjust learning rates and check data distribution
   python server.py --learning_rate 0.001 --iid True --verbose
   ```

5. **Dataset Download Failures:**
   ```bash
   # Clear cached data and retry
   rm -rf MNIST/ FashionMNIST/ cifar-10-batches-py/
   python server.py --dataset mnist  # Will re-download
   ```

### Debug Mode

Enable comprehensive debugging:
```bash
python server.py --debug --verbose --log_level DEBUG --save_results debug_results/
```

### Performance Profiling

Profile experiment execution:
```bash
python -m cProfile experiment_runner.py --test-mode > profile.txt
```

## 🚀 Quick Tips for Researchers

### Efficient Experimentation
1. **Start with test mode**: Use `--test-mode` to verify configurations
2. **Use intermediate saves**: Results saved every 5 experiments automatically
3. **Monitor logs**: Check `experiment_runner.log` for detailed progress
4. **Parallel analysis**: Load results while experiments are running

### Best Practices
1. **Version control**: Track experiment configurations and results
2. **Reproducibility**: Set random seeds for consistent results
3. **Documentation**: Document custom configurations and findings
4. **Resource management**: Monitor CPU/GPU usage during large studies


## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{federated_learning_framework_2025,
  title={A Comprehensive Federated Learning Framework with Attack Simulation and Systematic Evaluation},
  author={Andrea Ladiana},
  year={2025},
  url={https://github.com/andrea-ladiana/federated-learning-statistical-comparison},
  note={Framework for systematic evaluation of federated learning strategies under adversarial conditions}
}
```

## Acknowledgments

- **Flower Team**: For the excellent federated learning framework and baseline implementations
- **PyTorch Team**: For the robust deep learning library foundation
- **Research Community**: For the scientific papers and algorithms implemented in this framework
- **Open Source Contributors**: All developers who contributed to the underlying libraries and tools

### Scientific References

This framework implements algorithms from the following key papers:

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx) 
- Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
- Reddi et al. "Adaptive Federated Optimization" (FedAdam)
- Wang et al. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (FedNova)
- Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)
- El Mhamdi et al. "The Hidden Vulnerability of Distributed Learning in Byzantines" (Bulyan)
- Yin et al. "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates" (TrimmedMean)

---

## 📞 Support and Contact

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs and request features](https://github.com/andrea-ladiana/federated-learning-statistical-comparison/issues)
- **Research Collaboration**: Contact for academic partnerships and research collaborations
- **Educational Use**: Framework designed for FL courses and workshops

**Latest Version**: v4.0 (2025)  
**Compatibility**: Python 3.8+, PyTorch 1.9+, Flower 1.0+  
**Status**: Active development and maintenance  




---

*This framework is designed to accelerate federated learning research by providing a comprehensive, well-documented, and scientifically rigorous environment for experimentation and evaluation.*
