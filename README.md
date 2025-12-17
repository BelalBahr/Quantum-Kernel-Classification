# Quantum Kernel Classification Project

## Overview

This project implements, analyzes, and extends quantum kernel methods for machine learning. It builds quantum feature maps, simulates quantum kernels, and benchmarks quantum-enhanced classifiers against classical baselines.

The project aims to deepen understanding of how quantum properties can influence learning tasks and where quantum advantages might arise (or fail to).

## Project Structure

```
Quantum-Kernal-Classification/
├── project.py                              # Main implementation (all 5 parts)
├── check_progress.py                       # Progress monitoring script
├── PHM678_Quantum Kernel Classification Project.md  # Project requirements
├── PHM678_Quantum Kernel Classification Project.pdf # Project PDF
├── README.md                               # This file
├── quantum_kernel_basic_heatmap.png        # Basic feature map kernel visualization
├── quantum_kernel_zz_heatmap.png           # ZZ feature map kernel visualization
├── quantum_kernel_angle_encoding_heatmap.png # Angle encoding kernel visualization (generated)
└── svm_results_summary.txt                 # Performance results (generated)
```

## Requirements

### Python Packages

```bash
pip install numpy scikit-learn matplotlib qiskit qiskit-machine-learning qiskit-algorithms
```

**Required packages:**
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning (SVM, PCA, etc.)
- `matplotlib` - Visualization
- `qiskit` - Quantum computing framework
- `qiskit-machine-learning` - Quantum machine learning tools
- `qiskit-algorithms` - Quantum algorithms

## Dataset

- **Dataset**: Optical recognition of handwritten digits (from scikit-learn)
- **Binary Classification**: Digits 3 and 8
- **Original Features**: 64 (8×8 pixel images)
- **After PCA**: 16 features (for 4 qubits: 2^4 = 16)
- **Train/Test Split**: 80/20 (stratified)
- **Training Samples**: 100 (limited for faster computation)
- **Test Samples**: 30 (limited for faster computation)
- **Note**: Dataset is limited to reduce computational load while demonstrating concepts

## Implementation

### Part 1: Theoretical Background
- Review and summary of quantum kernel methods concepts
- Understanding quantum feature maps and kernel matrix construction

### Part 2: Implementing Quantum Feature Maps

Two quantum feature maps are implemented:

1. **Basic Quantum Feature Map**
   - Uses RX, RY, RZ rotations based on input features
   - 4 features per qubit (16 features total for 4 qubits)
   - Circuit depth: 4
   - Number of gates: 16

2. **ZZ Feature Map (Entangled)**
   - Uses RZ rotations + CNOT gates for entanglement
   - Creates correlations between features through quantum entanglement
   - 2 repetition layers for more expressivity
   - Circuit depth: 20
   - Number of gates: 56 (48 RZ + 8 CNOT)
   - **Note**: Uses linear chain + circular connectivity (4 product terms).
     A full ZZ map would encode all 6 pairwise interactions but is more expensive.

### Part 3: Quantum Kernel Computation

- Computes quantum kernel matrices using `FidelityQuantumKernel`
- Uses `StatevectorSampler` for exact quantum simulation
- Visualizes kernel matrices as heatmaps (50 training samples)
- Generates kernel matrices for SVM training

**Output Files:**
- `quantum_kernel_basic_heatmap.png` - Basic feature map kernel
- `quantum_kernel_zz_heatmap.png` - ZZ feature map kernel
- `quantum_kernel_angle_encoding_heatmap.png` - Angle encoding kernel

### Part 4: Classical SVM Training and Evaluation

Trains SVMs on quantum kernels and compares with classical kernels:

1. **Quantum Kernel SVMs:**
   - Basic Quantum Feature Map SVM
   - ZZ Quantum Feature Map SVM
   - Hyperparameter tuning (C parameter) with 5-fold CV

2. **Classical Kernel SVMs (for comparison):**
   - RBF Kernel SVM
   - Polynomial Kernel SVM
   - Linear Kernel SVM

**Output:**
- `svm_results_summary.txt` - Performance comparison table

### Part 5: Angle Embedding with QSVM

- Implements angle encoding feature map (RY rotations only)
- Encodes all 16 features (4 per qubit) using only RY rotations
- Simpler than basic feature map (no RX/RZ rotations)
- Trains QSVM (Quantum Support Vector Machine) using angle encoding
- Compares performance with Part 4 results
- Analyzes differences between encoding methods

## Usage

### Running the Full Project

```bash
python project.py
```

**Note:** Full execution takes approximately 10-15 minutes with the limited dataset (50 training + 10 test samples). This is much faster than using the full dataset while still demonstrating all concepts.

### Monitoring Progress

```bash
python check_progress.py
```

This script checks:
- Which output files have been created
- Whether Python processes are running
- Expected completion status

## Key Features

### Quantum Feature Maps

1. **Basic Feature Map**: Simple rotation-based encoding
   - Each feature encoded as rotation on qubit
   - Uses RX, RY, RZ rotations

2. **ZZ Feature Map**: Entangled encoding
   - RZ rotations + CNOT gates
   - Creates quantum correlations between features
   - More expressive but computationally intensive

3. **Angle Encoding**: Direct angle mapping
   - Uses only RY rotations (simpler than basic map)
   - Encodes all 16 features (4 per qubit)
   - Simpler circuit structure (no RX/RZ rotations, no entanglement)
   - Faster computation

### Quantum Kernel Computation

- Uses `FidelityQuantumKernel` for optimized computation
- Computes kernel matrices via quantum state fidelity
- Supports both visualization (50 samples) and full training (all samples)

### SVM Training

- Grid search hyperparameter tuning
- 5-fold cross-validation
- Performance metrics: accuracy, precision, recall, F1-score
- Confusion matrices for detailed analysis

## Output Files

### Generated During Execution

1. **Heatmaps** (PNG files):
   - `quantum_kernel_basic_heatmap.png`
   - `quantum_kernel_zz_heatmap.png`
   - `quantum_kernel_angle_encoding_heatmap.png`

2. **Results** (TXT file):
   - `svm_results_summary.txt` - Contains:
     - Test accuracies for all methods
     - Best hyperparameters
     - Performance comparison

## Performance Notes

- **Quantum Kernel Computation**: The slowest part of the pipeline
  - 50 training samples: ~5-10 minutes
  - 10 test samples: ~1-2 minutes
  - Each kernel entry requires quantum circuit simulation
  - Total runtime: ~10-15 minutes (much faster than full dataset)

- **SVM Training**: Fast once kernels are computed
  - Hyperparameter tuning: ~30 seconds
  - Training: < 10 seconds

## Project Deliverables

-Quantum feature maps (Basic + ZZ)  
-Quantum kernel computation with visualizations  
-SVM training and evaluation with classical comparison  
-Angle embedding QSVM implementation  

## Results Interpretation

The project compares:
- Quantum kernels vs. classical kernels
- Different quantum feature maps (Basic, ZZ, Angle Encoding)
- Performance metrics and computational complexity

Key insights:
- Quantum kernels can capture different feature relationships
- Entanglement (ZZ map) may improve expressivity
- Angle encoding provides simpler, faster alternative
- Performance depends on dataset and feature map choice

