# Federated Learning Framework with Flower

A comprehensive federated learning framework built on top of the Flower library, implementing multiple aggregation strategies, neural network models, datasets, and attack scenarios for research and educational purposes.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Usage](#command-line-usage)
- [Aggregation Strategies](#aggregation-strategies)
- [Neural Network Models](#neural-network-models)
- [Datasets](#datasets)
- [Attack Implementations](#attack-implementations)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Baselines and Benchmarks](#baselines-and-benchmarks)
- [Contributing](#contributing)
- [Documentation](#documentation)

## Overview

This federated learning framework provides a complete research and educational environment for experimenting with federated learning algorithms. It includes:

- **19 federated learning strategies** (10 custom implementations + 8 Flower baselines + 1 additional)
- **6 neural network models** optimized for different tasks
- **3 popular datasets** (MNIST, Fashion-MNIST, CIFAR-10)
- **6 attack types** for security research
- **Comprehensive baseline integrations** from the Flower ecosystem
- **Educational documentation** with scientific references

## Features

### Core Capabilities
- ✅ Multi-strategy federated learning simulation
- ✅ Robust aggregation algorithms (Byzantine-fault tolerant)
- ✅ Attack simulation and defense evaluation
- ✅ Non-IID data distribution support
- ✅ Comprehensive model zoo
- ✅ Educational documentation with scientific papers

### Research Features
- ✅ Byzantine fault tolerance evaluation
- ✅ Adversarial attack simulations
- ✅ Statistical heterogeneity handling
- ✅ Communication efficiency optimization
- ✅ Convergence analysis tools

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd v3
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
conda env create -f environment.yml
conda activate federated-learning
```

5. **Install Flower (if needed):**
```bash
python install_flower.py
```

## Quick Start

### Basic Federated Learning

Run a simple federated learning simulation with FedAvg strategy:

```bash
python server.py --strategy fedavg --model CNNNet --dataset mnist --num_clients 10 --num_rounds 10
```

### With Attack Simulation

Run federated learning with adversarial attacks:

```bash
python run_with_attacks.py --strategy fedavg --model CNNNet --dataset cifar10 --attack_type label_flipping --malicious_clients 2
```

## Command-Line Usage

### Server Configuration (`server.py`)

The main server script supports the following parameters:

#### Core Parameters
```bash
python server.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--strategy` | str | `fedavg` | Aggregation strategy to use |
| `--model` | str | `CNNNet` | Neural network model |
| `--dataset` | str | `mnist` | Dataset for training |
| `--num_clients` | int | `10` | Number of participating clients |
| `--num_rounds` | int | `10` | Number of federated learning rounds |
| `--fraction_fit` | float | `1.0` | Fraction of clients for training |
| `--fraction_evaluate` | float | `1.0` | Fraction of clients for evaluation |
| `--min_fit_clients` | int | `2` | Minimum clients for training round |
| `--min_evaluate_clients` | int | `2` | Minimum clients for evaluation |
| `--min_available_clients` | int | `2` | Minimum clients to start |

#### Advanced Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--learning_rate` | float | `0.01` | Client learning rate |
| `--local_epochs` | int | `1` | Local training epochs per round |
| `--batch_size` | int | `32` | Training batch size |
| `--test_batch_size` | int | `1000` | Evaluation batch size |
| `--iid` | bool | `True` | Use IID data distribution |
| `--verbose` | bool | `False` | Enable verbose logging |

#### Strategy-Specific Parameters

**FedProx:**
```bash
--proximal_mu 0.01  # Proximal term coefficient
```

**FedAdam:**
```bash
--eta 1e-3          # Server learning rate
--eta_l 1e-1        # Local learning rate  
--beta_1 0.9        # First moment decay
--beta_2 0.99       # Second moment decay
--tau 1e-3          # Control parameter
```

**SCAFFOLD:**
```bash
--eta_l 1.0         # Local learning rate
--eta_g 1.0         # Global learning rate
```

**Byzantine-Fault Tolerant Strategies:**
```bash
--num_malicious 2   # Number of malicious clients (for Krum, Bulyan, TrimmedMean)
```

### Attack Simulation (`run_with_attacks.py`)

Run federated learning with adversarial scenarios:

```bash
python run_with_attacks.py [OPTIONS]
```

#### Attack Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--attack_type` | str | `None` | Type of attack to simulate |
| `--malicious_clients` | int | `0` | Number of malicious clients |
| `--attack_intensity` | float | `0.5` | Attack strength/intensity |
| `--attack_rounds` | list | `[]` | Specific rounds for attacks |

#### Attack-Specific Parameters

**Label Flipping:**
```bash
--flip_probability 0.1      # Probability of label flip
--target_class 7            # Target class for flipping
```

**Gradient Flipping:**
```bash
--flip_factor -1.0          # Gradient multiplication factor
```

**Noise Injection:**
```bash
--noise_std 0.1             # Standard deviation of Gaussian noise
--noise_type gaussian       # Type of noise (gaussian, uniform)
```

**Data Asymmetry:**
```bash
--asymmetry_factor 0.8      # Level of data asymmetry
--missing_classes [0,1,2]   # Classes to exclude from clients
```

**Client Failure:**
```bash
--failure_probability 0.1   # Probability of client failure
--failure_type random       # Type of failure (random, targeted)
```

**Missed Class:**
```bash
--excluded_classes [9]      # Classes to exclude from specific clients
--affected_clients [0,1]    # Clients affected by missing classes
```

## Aggregation Strategies

The framework implements 19 different federated learning strategies:

### 1. Custom Implementations (10 strategies)

#### **FedAvg (Federated Averaging)**
- **Paper:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Description:** Standard federated averaging algorithm
- **Usage:** `--strategy fedavg`
- **Parameters:** Standard FL parameters

#### **FedProx (Federated Proximal)**
- **Paper:** Li et al., "Federated Optimization in Heterogeneous Networks"
- **Description:** Adds proximal term to handle heterogeneity
- **Usage:** `--strategy fedprox --proximal_mu 0.01`
- **Key Parameter:** `proximal_mu` - controls regularization strength

#### **FedAdam (Federated Adam)**
- **Paper:** Reddi et al., "Adaptive Federated Optimization"
- **Description:** Server-side adaptive optimization
- **Usage:** `--strategy fedadam --eta 1e-3 --beta_1 0.9 --beta_2 0.99`
- **Parameters:**
  - `eta`: Server learning rate
  - `eta_l`: Local learning rate
  - `beta_1`, `beta_2`: Momentum parameters
  - `tau`: Control parameter

#### **SCAFFOLD**
- **Paper:** Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
- **Description:** Uses control variates to reduce client drift
- **Usage:** `--strategy scaffold --eta_l 1.0 --eta_g 1.0`
- **Parameters:**
  - `eta_l`: Local learning rate
  - `eta_g`: Global learning rate

#### **FedNova**
- **Paper:** Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
- **Description:** Normalized averaging for heterogeneous objectives
- **Usage:** `--strategy fednova`

#### **FedAvgM (Federated Averaging with Momentum)**
- **Paper:** Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification"
- **Description:** Server-side momentum for faster convergence
- **Usage:** `--strategy fedavgm --server_momentum 0.9`

### 2. Byzantine-Fault Tolerant Strategies (3 strategies)

#### **Krum**
- **Paper:** Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- **Description:** Selects most representative client update
- **Usage:** `--strategy krum --num_malicious 2`
- **Tolerates:** Up to `(n-f-2)/2` malicious clients where `n` is total clients and `f` is malicious

#### **Bulyan**
- **Paper:** El Mhamdi et al., "The Hidden Vulnerability of Distributed Learning in Byzantines"
- **Description:** Combines Krum with trimmed mean
- **Usage:** `--strategy bulyan --num_malicious 2`
- **Tolerates:** Up to `(n-2f)/3` malicious clients

#### **Trimmed Mean**
- **Paper:** Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
- **Description:** Removes extreme values before averaging
- **Usage:** `--strategy trimmed_mean --num_malicious 2`
- **Tolerates:** Up to `n/2 - 1` malicious clients

### 3. Flower Baseline Strategies (8 strategies)

Integrated from the Flower ecosystem:

- **FedOpt:** Federated optimization framework
- **FedYogi:** Adaptive federated optimization with Yogi
- **QFedAvg:** Fair federated learning
- **FaultTolerantFedAvg:** Fault-tolerant federated averaging
- **FastAndSlow:** Mixed-speed client handling
- **FedTrimmedAvg:** Trimmed averaging implementation
- **FedMedian:** Median-based aggregation
- **DPFedAvgFixed:** Differential privacy with fixed noise

### Strategy Selection Guide

| Use Case | Recommended Strategy | Key Benefits |
|----------|---------------------|--------------|
| **Standard FL** | FedAvg | Simple, well-established |
| **Non-IID Data** | FedProx, SCAFFOLD | Handles heterogeneity |
| **Fast Convergence** | FedAdam, FedAvgM | Adaptive optimization |
| **Byzantine Attacks** | Krum, Bulyan, TrimmedMean | Robust to malicious clients |
| **Communication Efficiency** | FedNova | Fewer communication rounds |
| **Privacy-Preserving** | DPFedAvgFixed | Differential privacy |

## Neural Network Models

The framework provides 6 neural network models optimized for different scenarios:

### 1. **Net (Simple MLP)**
- **Architecture:** 2-layer Multi-Layer Perceptron
- **Input:** 28×28 (784 features)
- **Layers:** 784 → 128 → 64 → 10
- **Use Case:** MNIST baseline, educational purposes
- **Parameters:** ~101K

### 2. **CNNNet (Convolutional Neural Network)**
- **Architecture:** CNN with 2 conv layers + 2 FC layers
- **Input:** 28×28×1 or 32×32×3
- **Layers:** Conv(32) → Conv(64) → FC(128) → FC(10)
- **Use Case:** Image classification, general purpose
- **Parameters:** ~1.2M

### 3. **TinyMNIST (Lightweight CNN)**
- **Architecture:** Minimal CNN for MNIST
- **Input:** 28×28×1
- **Layers:** Conv(8) → Conv(16) → FC(10)
- **Use Case:** Resource-constrained environments
- **Parameters:** ~25K

### 4. **MinimalCNN (Ultra-Light CNN)**
- **Architecture:** Extremely lightweight CNN
- **Input:** 28×28×1
- **Layers:** Conv(4) → Conv(8) → FC(10)
- **Use Case:** Edge devices, IoT scenarios
- **Parameters:** ~8K

### 5. **MiniResNet20 (Residual Network)**
- **Architecture:** Scaled-down ResNet with 20 layers
- **Input:** 32×32×3
- **Layers:** ResNet blocks with skip connections
- **Use Case:** CIFAR-10, complex image tasks
- **Parameters:** ~270K

### 6. **OptAEGV3 (Optimized Architecture)**
- **Architecture:** Advanced CNN with depthwise separable convolutions
- **Input:** Variable (adaptive)
- **Layers:** Efficient convolutions + global pooling
- **Use Case:** Efficient inference, mobile deployment
- **Parameters:** ~150K

### Model Selection Guide

| Dataset | Recommended Model | Reasoning |
|---------|------------------|-----------|
| **MNIST** | TinyMNIST, Net | Simple digits, lightweight models sufficient |
| **Fashion-MNIST** | CNNNet, MinimalCNN | More complex patterns, need conv layers |
| **CIFAR-10** | MiniResNet20, OptAEGV3 | Complex images, require deeper networks |

### Model Specifications

| Model | Input Size | Parameters | FLOPS | Memory (MB) | Training Time |
|-------|------------|------------|-------|-------------|---------------|
| Net | 28×28 | 101K | 0.2M | 0.4 | Fast |
| CNNNet | 28×28/32×32 | 1.2M | 15M | 4.8 | Medium |
| TinyMNIST | 28×28 | 25K | 1.2M | 0.1 | Very Fast |
| MinimalCNN | 28×28 | 8K | 0.5M | <0.1 | Very Fast |
| MiniResNet20 | 32×32 | 270K | 40M | 1.1 | Medium |
| OptAEGV3 | Variable | 150K | 8M | 0.6 | Fast |

## Datasets

The framework supports 3 popular computer vision datasets:

### 1. **MNIST (Modified National Institute of Standards and Technology)**
- **Description:** Handwritten digits (0-9)
- **Size:** 70,000 images (60K train + 10K test)
- **Resolution:** 28×28 grayscale
- **Classes:** 10 (digits 0-9)
- **Usage:** `--dataset mnist`
- **Characteristics:**
  - Simple, well-balanced dataset
  - Low resolution, single channel
  - Good for proof-of-concept and debugging
  - IID and Non-IID partitioning supported

### 2. **Fashion-MNIST**
- **Description:** Fashion items (clothing, shoes, bags)
- **Size:** 70,000 images (60K train + 10K test)
- **Resolution:** 28×28 grayscale
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Usage:** `--dataset fashion_mnist`
- **Characteristics:**
  - More complex than MNIST
  - Similar structure to MNIST (drop-in replacement)
  - Better for evaluating model generalization
  - Realistic federated learning scenarios

### 3. **CIFAR-10 (Canadian Institute For Advanced Research)**
- **Description:** Natural color images
- **Size:** 60,000 images (50K train + 10K test)
- **Resolution:** 32×32 RGB (3 channels)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Usage:** `--dataset cifar10`
- **Characteristics:**
  - Most challenging dataset
  - Color images with complex backgrounds
  - Requires deeper networks
  - Realistic for computer vision tasks

### Data Distribution Options

#### **IID (Independent and Identically Distributed)**
```bash
--iid True
```
- Each client receives random samples from all classes
- Balanced distribution across clients
- Ideal scenario (rarely realistic)

#### **Non-IID (Non-Independent and Identically Distributed)**
```bash
--iid False
```
- Heterogeneous data distribution
- Each client may have different class distributions
- More realistic federated learning scenario
- Tests algorithm robustness

### Dataset Partitioning Strategies

1. **Random IID:** Equal random sampling across all clients
2. **Class-based Non-IID:** Each client gets samples from subset of classes
3. **Quantity-based Non-IID:** Unequal sample sizes per client
4. **Dirichlet Non-IID:** Continuous control over heterogeneity level

### Dataset Statistics

| Dataset | Train Size | Test Size | Image Size | Channels | Classes | Complexity |
|---------|------------|-----------|------------|----------|---------|------------|
| MNIST | 60,000 | 10,000 | 28×28 | 1 | 10 | Low |
| Fashion-MNIST | 60,000 | 10,000 | 28×28 | 1 | 10 | Medium |
| CIFAR-10 | 50,000 | 10,000 | 32×32 | 3 | 10 | High |

## Attack Implementations

The framework includes 6 different attack types for security research:

### 1. **Label Flipping Attack**
- **File:** `attacks/label_flipping.py`
- **Description:** Malicious clients flip labels of training data
- **Usage:** `--attack_type label_flipping --flip_probability 0.1 --target_class 7`
- **Parameters:**
  - `flip_probability`: Probability of flipping a label (0.0-1.0)
  - `target_class`: Target class for directed attacks (optional)
- **Impact:** Degrades model accuracy on specific classes
- **Detection:** Monitor per-class accuracy degradation

### 2. **Gradient Flipping Attack**
- **File:** `attacks/gradient_flipping.py`
- **Description:** Malicious clients send inverted gradients
- **Usage:** `--attack_type gradient_flipping --flip_factor -1.0`
- **Parameters:**
  - `flip_factor`: Multiplication factor for gradients (negative values flip)
- **Impact:** Severely disrupts convergence
- **Detection:** Monitor gradient norms and directions

### 3. **Noise Injection Attack**
- **File:** `attacks/noise_injection.py`
- **Description:** Adds random noise to client updates
- **Usage:** `--attack_type noise_injection --noise_std 0.1 --noise_type gaussian`
- **Parameters:**
  - `noise_std`: Standard deviation of noise
  - `noise_type`: Type of noise (gaussian, uniform, laplacian)
- **Impact:** Slows convergence, reduces final accuracy
- **Detection:** Statistical analysis of update distributions

### 4. **Data Asymmetry Attack**
- **File:** `attacks/data_asymmetry.py`
- **Description:** Creates extreme data heterogeneity
- **Usage:** `--attack_type data_asymmetry --asymmetry_factor 0.8`
- **Parameters:**
  - `asymmetry_factor`: Level of data skewness (0.0-1.0)
  - `missing_classes`: Specific classes to exclude
- **Impact:** Biases global model toward specific classes
- **Detection:** Monitor class distribution across clients

### 5. **Client Failure Attack**
- **File:** `attacks/client_failure.py`
- **Description:** Simulates client dropouts and failures
- **Usage:** `--attack_type client_failure --failure_probability 0.1 --failure_type random`
- **Parameters:**
  - `failure_probability`: Probability of client failure per round
  - `failure_type`: Pattern of failures (random, targeted, gradual)
- **Impact:** Reduces available data for training
- **Detection:** Monitor client participation rates

### 6. **Missed Class Attack**
- **File:** `attacks/missed_class.py`
- **Description:** Specific clients never see certain classes
- **Usage:** `--attack_type missed_class --excluded_classes [9] --affected_clients [0,1]`
- **Parameters:**
  - `excluded_classes`: List of classes to exclude
  - `affected_clients`: List of client IDs to affect
- **Impact:** Creates systematic bias in learning
- **Detection:** Analyze per-client class distributions

### Attack Configuration

Attacks are configured through the `attack_config.py` file, which provides:

- **Attack Factory:** Centralized attack instantiation
- **Parameter Validation:** Ensures valid attack parameters
- **Attack Scheduling:** Controls when attacks occur
- **Intensity Control:** Manages attack strength over time

### Defense Evaluation

The framework supports evaluation of various defense mechanisms:

1. **Robust Aggregation:** Use Byzantine-fault tolerant strategies
2. **Outlier Detection:** Identify anomalous client updates
3. **Differential Privacy:** Add noise to protect against inference attacks
4. **Client Reputation:** Track client behavior over time
5. **Secure Aggregation:** Cryptographic protection of updates

### Attack Impact Metrics

| Attack Type | Primary Impact | Secondary Impact | Detection Difficulty |
|-------------|----------------|------------------|---------------------|
| Label Flipping | Accuracy Loss | Class Bias | Medium |
| Gradient Flipping | Convergence Failure | Training Instability | Easy |
| Noise Injection | Slow Convergence | Accuracy Degradation | Hard |
| Data Asymmetry | Model Bias | Unfairness | Medium |
| Client Failure | Reduced Data | Communication Issues | Easy |
| Missed Class | Class Imbalance | Systematic Bias | Hard |

## Project Structure

```
v3/
├── README.md                    # This comprehensive guide
├── requirements.txt             # Python dependencies
├── environment.yml             # Conda environment
├── setup.py                    # Package installation
│
├── server.py                   # Main server entry point
├── client.py                   # Client implementation
├── run_with_attacks.py         # Attack simulation script
├── strategies.py               # Strategy factory
├── models.py                   # Model imports (compatibility)
├── attack_config.py            # Attack configuration
├── utils.py                    # Utility functions
│
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── simple.py              # Net (MLP)
│   ├── cnn.py                 # CNNNet
│   ├── minimal_cnn.py         # TinyMNIST, MinimalCNN
│   ├── miniresnet20.py        # MiniResNet20
│   └── optaegv3.py            # OptAEGV3
│
├── strategy/                   # Aggregation strategies
│   ├── __init__.py
│   ├── fedavg.py              # FedAvg implementation
│   ├── fedprox.py             # FedProx implementation
│   ├── fedadam.py             # FedAdam implementation
│   ├── scaffold.py            # SCAFFOLD implementation
│   ├── fednova.py             # FedNova implementation
│   ├── fedavgm.py             # FedAvgM implementation
│   ├── krum.py                # Krum (Byzantine-tolerant)
│   ├── bulyan.py              # Bulyan (Byzantine-tolerant)
│   ├── trimmed_mean.py        # TrimmedMean (Byzantine-tolerant)
│   └── baseline_wrappers.py   # Flower baseline integrations
│
├── attacks/                    # Attack implementations
│   ├── __init__.py
│   ├── label_flipping.py      # Label flipping attack
│   ├── gradient_flipping.py   # Gradient flipping attack
│   ├── noise_injection.py     # Noise injection attack
│   ├── data_asymmetry.py      # Data asymmetry attack
│   ├── client_failure.py      # Client failure simulation
│   └── missed_class.py        # Missed class attack
│
├── baselines/                  # Flower baseline integrations
│   ├── niid_bench/            # Non-IID benchmarking
│   ├── tamuna/                # TAMUNA algorithm
│   ├── statavg/               # Statistical averaging
│   ├── hfedxgboost/          # Federated XGBoost
│   └── [other baselines]/    # Additional implementations
│
├── docs/                       # Scientific papers and documentation
│   ├── FedAvg.pdf
│   ├── FedProx.pdf
│   ├── SCAFFOLD.pdf
│   ├── FedAdam.pdf
│   ├── Krum.pdf
│   ├── Bulyan.pdf
│   └── [other papers]
│
└── reports/                    # Analysis reports
    ├── STRATEGY_COMPLIANCE_REPORT.md
    └── BASELINE_INTEGRATION_REPORT.md
```

## Advanced Usage

### Custom Strategy Implementation

To implement a new aggregation strategy:

1. Create a new file in `strategy/` directory
2. Inherit from appropriate base class
3. Implement required methods
4. Register in `strategies.py`

Example:
```python
from flwr.server.strategy import Strategy
from typing import List, Tuple, Optional
from flwr.common import Parameters, FitRes

class CustomStrategy(Strategy):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Your aggregation logic here
        pass
```

### Custom Attack Implementation

To implement a new attack:

1. Create a new file in `attacks/` directory
2. Implement attack class with `apply_attack` method
3. Register in `attack_config.py`

Example:
```python
class CustomAttack:
    def __init__(self, attack_params):
        self.params = attack_params
    
    def apply_attack(self, data, labels):
        # Your attack logic here
        return modified_data, modified_labels
```

### Hyperparameter Tuning

Use the built-in parameter sweeps:

```bash
# Grid search over learning rates
python server.py --strategy fedavg --learning_rate 0.001,0.01,0.1

# Test multiple strategies
python run_with_attacks.py --strategy fedavg,fedprox,scaffold --attack_type label_flipping
```

### Logging and Monitoring

Enable detailed logging:

```bash
python server.py --verbose --log_level DEBUG --save_results results/
```

### Multi-GPU Support

For large-scale experiments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python server.py --num_clients 100 --parallel_training
```

## Baselines and Benchmarks

The framework integrates multiple baseline implementations from the Flower ecosystem:

### Available Baselines

1. **NIID-Bench:** Comprehensive non-IID evaluation
2. **TAMUNA:** Communication-efficient federated learning
3. **StatAvg:** Statistical averaging methods
4. **HFedXGBoost:** Federated gradient boosting
5. **FedBN:** Federated batch normalization
6. **FedProx:** Advanced FedProx implementation
7. **FedNova:** Official FedNova baseline
8. **And many more...**

### Running Baselines

Each baseline has its own configuration and can be run independently:

```bash
# Run NIID-Bench
cd baselines/niid_bench
python -m niid_bench.main --config-name fedavg_base

# Run TAMUNA
cd baselines/tamuna
python -m tamuna.main

# Run HFedXGBoost
cd baselines/hfedxgboost
python -m hfedxgboost.main --config-name Centralized_Baseline
```

### Benchmark Results

See `reports/BASELINE_INTEGRATION_REPORT.md` for comprehensive benchmark results and comparisons.

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

### Scientific Papers

The `docs/` directory contains original papers for all implemented algorithms:

- **FedAvg.pdf:** Original federated averaging paper
- **FedProx.pdf:** Proximal federated optimization
- **SCAFFOLD.pdf:** Controlled averaging for FL
- **FedAdam.pdf:** Adaptive federated optimization
- **Krum.pdf:** Byzantine-tolerant aggregation
- **Bulyan.pdf:** Enhanced Byzantine tolerance
- **TrimmedMean.pdf:** Robust aggregation method

### Reports

- **STRATEGY_COMPLIANCE_REPORT.md:** Analysis of strategy implementations
- **BASELINE_INTEGRATION_REPORT.md:** Baseline integration status and results

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
   - Reduce batch size or model size
   - Use CPU training for debugging

2. **Client Connection Issues:**
   - Check network configuration
   - Verify client resources

3. **Convergence Problems:**
   - Adjust learning rates
   - Check data distribution
   - Verify attack parameters

### Debug Mode

Enable debug mode for detailed logs:
```bash
python server.py --debug --verbose --log_level DEBUG
```

### Performance Profiling

Profile your experiments:
```bash
python -m cProfile server.py --strategy fedavg > profile.txt
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{federated_learning_framework,
  title={Comprehensive Federated Learning Framework with Flower},
  year={2024},
  url={https://github.com/your-repo/federated-learning}
}
```

## Acknowledgments

- **Flower Team:** For the excellent federated learning framework
- **PyTorch Team:** For the deep learning library
- **Research Community:** For the scientific papers and algorithms implemented
- **Contributors:** All developers who contributed to this project

---

For more information, support, or to report issues, please visit our [GitHub repository](https://github.com/your-repo/federated-learning) or contact the development team.
