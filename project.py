import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt

# Load the Optical recognition of handwritten digits dataset
digits = load_digits()

# Extract features and labels
X = digits.data
y = digits.target

# Filter to only digits 3 and 8
mask = (y == 3) | (y == 8)
X_filtered = X[mask]
y_filtered = y[mask]

# Convert labels to binary immediately (3 -> 0, 8 -> 1) for consistency
y_filtered_binary = (y_filtered == 8).astype(int)

print(f"Original dataset shape: {X.shape}")
print(f"Filtered dataset shape: {X_filtered.shape}")
print(f"Number of digit 3 samples: {np.sum(y_filtered == 3)}")
print(f"Number of digit 8 samples: {np.sum(y_filtered == 8)}")
print(f"Unique labels in filtered dataset: {np.unique(y_filtered)}")

# Split train/test (80/20)
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X_filtered, y_filtered_binary, test_size=0.2, random_state=42, stratify=y_filtered_binary
)

# Use sufficient samples for meaningful results
#Using 100/30
n_train_samples = 100
n_test_samples = 30

# Limit dataset size for faster computation while maintaining statistical validity
if len(X_train) > n_train_samples:
    X_train = X_train[:n_train_samples]
    y_train_binary = y_train_binary[:n_train_samples]
if len(X_test) > n_test_samples:
    X_test = X_test[:n_test_samples]
    y_test_binary = y_test_binary[:n_test_samples]

print(f"\nTrain set shape: {X_train.shape} ({len(X_train)} samples)")
print(f"Test set shape: {X_test.shape} ({len(X_test)} samples)")
print(f"Number of class 0 (digit 3) in train: {np.sum(y_train_binary == 0)}")
print(f"Number of class 1 (digit 8) in train: {np.sum(y_train_binary == 1)}")
print(f"Number of class 0 (digit 3) in test: {np.sum(y_test_binary == 0)}")
print(f"Number of class 1 (digit 8) in test: {np.sum(y_test_binary == 1)}")

# Apply PCA to training set (4 qubits = 16 features)
# IMPORTANT: Fit PCA only on training set to prevent data leakage
# Then transform both training and test sets using the fitted PCA
n_qubits = 4
n_features = 2 ** n_qubits  # 16 features for 4 qubits
pca = PCA(n_components=n_features)
# Fit and transform training set
X_train_pca = pca.fit_transform(X_train)
# Transform test set using the PCA fitted on training data (prevents data leakage)
X_test_pca = pca.transform(X_test)

# Check variance captured
variance_captured = np.sum(pca.explained_variance_ratio_)
print(f"\nPCA Results (4 qubits = {n_features} features):")
print(f"Original feature dimension: {X_train.shape[1]}")
print(f"PCA feature dimension: {X_train_pca.shape[1]}")
print(f"Explained variance ratio (all {n_features} components): {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {variance_captured:.4f}")
if variance_captured < 0.70:
    print(f"WARNING: Low variance captured ({variance_captured:.3f}), consider adjusting n_components")
print(f"\nTrain set after PCA shape: {X_train_pca.shape}")
print(f"Test set after PCA shape: {X_test_pca.shape}")

# Scale features to rotation angle range [-pi, pi] for meaningful rotations
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)
print(f"\nFeatures scaled to rotation angle range: [-π, π]")
print(f"Train set after scaling: min={X_train_pca.min():.4f}, max={X_train_pca.max():.4f}")
print(f"Test set after scaling: min={X_test_pca.min():.4f}, max={X_test_pca.max():.4f}")

# Basic Quantum Feature Map: Encode each feature as rotation on one qubit
def create_quantum_feature_map(n_qubits, n_features):
    """
    Create a basic quantum feature map where each feature is encoded as a rotation on a qubit.
    For 16 features and 4 qubits, we encode 4 features per qubit using RX, RY, RZ rotations.
    """
    # Create parameter vector for features
    features = ParameterVector('x', n_features)
    
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    # Encode features: 4 features per qubit
    features_per_qubit = n_features // n_qubits
    
    for qubit in range(n_qubits):
        # Get the features for this qubit
        start_idx = qubit * features_per_qubit
        
        # Apply rotations: RX, RY, RZ for first 3 features, RX again for 4th
        if start_idx < n_features:
            qc.rx(features[start_idx], qubit)
        if start_idx + 1 < n_features:
            qc.ry(features[start_idx + 1], qubit)
        if start_idx + 2 < n_features:
            qc.rz(features[start_idx + 2], qubit)
        if start_idx + 3 < n_features:
            qc.rx(features[start_idx + 3], qubit)
    
    return qc, features

# ZZ Feature Map: Entangled feature map with ZZ interactions
def create_zz_feature_map(n_qubits, n_features, reps=2):
    """
    Create a proper ZZ feature map with actual ZZ interactions.
    This encodes feature products through ZZ interactions, not just entanglement.
    
    Structure:
    - Hadamard layer
    - Single-qubit RZ rotations encoding features (no factor of 2, features already scaled to [-π, π])
    - ZZ interactions encoding feature products (core of ZZ feature map)
    
    Note: Features are already scaled to [-π, π] range, so we use them directly
    without the factor of 2 that's sometimes used in standard ZZ feature maps.
    
    LIMITATION: This implementation uses linear chain + circular connectivity,
    encoding 4 product terms (f₀·f₄, f₄·f₈, f₈·f₁₂, f₁₂·f₀).
    A full ZZ feature map would encode all ½n(n-1) = 6 pairwise interactions
    for 4 qubits, but would be ~50% more computationally expensive.
    This limited version demonstrates the core concept while remaining practical.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of features to encode
        reps: Number of repetition layers (default: 2)
    """
    # Create parameter vector for features
    features = ParameterVector('x', n_features)
    
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    # Encode features: 4 features per qubit
    features_per_qubit = n_features // n_qubits
    
    for rep in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            qc.h(i)
        
        # Single-qubit rotations encoding features
        for qubit in range(n_qubits):
            start_idx = qubit * features_per_qubit
            # Apply RZ rotations for each feature on this qubit
            for i in range(features_per_qubit):
                if start_idx + i < n_features:
                    qc.rz(features[start_idx + i], qubit)
        
        # ZZ interactions with feature products (core of ZZ feature map)
        # This encodes feature interactions through product terms
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
            # Encode feature product as rotation (actual ZZ interaction)
            feature_idx_i = i * features_per_qubit
            feature_idx_j = (i+1) * features_per_qubit
            if feature_idx_i < n_features and feature_idx_j < n_features:
                qc.rz(features[feature_idx_i] * features[feature_idx_j], i+1)
            qc.cx(i, i+1)
        
        # Circular connectivity (last qubit -> first qubit)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)
            feature_idx_last = (n_qubits - 1) * features_per_qubit
            feature_idx_first = 0
            if feature_idx_last < n_features:
                qc.rz(features[feature_idx_last] * features[feature_idx_first], 0)
            qc.cx(n_qubits - 1, 0)
    
    return qc, features

# Create the basic quantum feature map
quantum_feature_map, feature_params = create_quantum_feature_map(n_qubits, n_features)

print(f"\nQuantum Feature Map:")
print(f"Number of qubits: {n_qubits}")
print(f"Number of features: {n_features}")
print(f"Features per qubit: {n_features // n_qubits}")
print(f"Circuit depth: {quantum_feature_map.depth()}")
print(f"Number of gates: {len(quantum_feature_map.data)}")

# Show circuit structure
print(f"\nCircuit gates:")
for i, instruction in enumerate(quantum_feature_map.data):
    gate_name = instruction.operation.name
    qubits = [quantum_feature_map.find_bit(q)[0] for q in instruction.qubits]
    params = [p.name if hasattr(p, 'name') else str(p) for p in instruction.operation.params]
    print(f"  Gate {i+1}: {gate_name} on qubit(s) {qubits}, params: {params}")

# Test encoding a single sample
print(f"\nTesting feature map with first training sample:")
sample = X_train_pca[0]
print(f"Sample features (first 4): {sample[:4]}")
print(f"Sample shape: {sample.shape}")

# Bind parameters to create a concrete circuit for this sample
qc_sample = quantum_feature_map.assign_parameters(dict(zip(feature_params, sample)))
print(f"\nCircuit with sample features bound:")
print(f"  Circuit depth: {qc_sample.depth()}")
print(f"  Number of gates: {len(qc_sample.data)}")
print(f"  First few gates:")
for i, instruction in enumerate(qc_sample.data[:4]):
    gate_name = instruction.operation.name
    qubits = [qc_sample.find_bit(q)[0] for q in instruction.qubits]
    params = [f"{p:.4f}" for p in instruction.operation.params]
    print(f"    Gate {i+1}: {gate_name} on qubit(s) {qubits}, angle: {params}")

# Visualize circuit for a few input samples (Part 2 requirement)
print("\nVisualizing circuits for a few input samples...")
for i in range(min(3, len(X_train_pca))):  # Visualize first 3 samples
    sample = X_train_pca[i]
    qc_vis = quantum_feature_map.assign_parameters(dict(zip(feature_params, sample)))
    
    # Save circuit diagram
    try:
        fig = qc_vis.draw(output='mpl', style='iqp', scale=0.8)
        plt.savefig(f'circuit_basic_feature_map_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved circuit visualization for sample {i+1}: circuit_basic_feature_map_sample_{i+1}.png")
    except Exception as e:
        # Properly handle exception
        print(f"  Skipping visualization for sample {i+1}: {e}")

# Create the ZZ (entangled) feature map
print(f"\n{'='*60}")
print("ZZ Feature Map (Entangled Feature Map)")
print(f"{'='*60}")

zz_feature_map, zz_feature_params = create_zz_feature_map(n_qubits, n_features, reps=2)

print(f"\nZZ Feature Map:")
print(f"Number of qubits: {n_qubits}")
print(f"Number of features: {n_features}")
print(f"Repetitions: 2")
print(f"Circuit depth: {zz_feature_map.depth()}")
print(f"Number of gates: {len(zz_feature_map.data)}")

# Show circuit structure
print(f"\nCircuit gates (first 12):")
for i, instruction in enumerate(zz_feature_map.data[:12]):
    gate_name = instruction.operation.name
    qubits = [zz_feature_map.find_bit(q)[0] for q in instruction.qubits]
    if instruction.operation.params:
        params = [p.name if hasattr(p, 'name') else str(p) for p in instruction.operation.params]
        print(f"  Gate {i+1}: {gate_name} on qubit(s) {qubits}, params: {params}")
    else:
        print(f"  Gate {i+1}: {gate_name} on qubit(s) {qubits}")

# Count gate types
gate_counts = {}
for instruction in zz_feature_map.data:
    gate_name = instruction.operation.name
    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

print(f"\nGate counts:")
for gate, count in gate_counts.items():
    print(f"  {gate}: {count}")

# ============================================================================
# Part 2: Discussion of Feature Map Design Intuition
# ============================================================================
print(f"\n{'='*60}")
print("Part 2: Feature Map Design Intuition and Discussion")
print(f"{'='*60}")

print("\n1. Basic Quantum Feature Map:")
print("   Design: Uses RX, RY, RZ rotations to encode features")
print("   Intuition:")
print("   - Each feature is encoded as a rotation angle on a qubit")
print("   - RX, RY, RZ rotations provide full coverage of the Bloch sphere")
print("   - No entanglement between qubits - features are encoded independently")
print("   - Expressivity: Moderate - can represent individual feature values")
print("   - Use case: Simple, interpretable encoding for basic feature representation")

print("\n2. ZZ Feature Map (Entangled):")
print("   Design: Uses Hadamard, RZ rotations, and ZZ interactions with feature products")
print("   Intuition:")
print("   - Hadamard gates prepare superposition states")
print("   - RZ rotations encode individual features as phase rotations")
print("   - ZZ interactions encode feature PRODUCTS (x[i] * x[j]) as rotations")
print("   - This captures pairwise feature interactions, not just entanglement")
print("   - Expressivity: High - can capture complex feature interactions and correlations")
print("   - Use case: When feature relationships and pairwise interactions are important")
print("   - Trade-off: More computationally expensive due to ZZ interactions")
print("   - LIMITATION: Uses linear chain + circular connectivity (4 product terms)")
print("     Full ZZ would encode all 6 pairwise interactions but is ~50% more expensive")
print("     This demonstrates the concept while remaining computationally practical")

print("\n3. Key Differences:")
print("   - Basic map: Independent feature encoding, simpler circuit")
print("   - ZZ map: Correlated feature encoding via entanglement, more complex")
print("   - Basic map: Faster computation, easier to interpret")
print("   - ZZ map: Potentially better for complex patterns, slower computation")

# Test encoding a single sample with ZZ feature map
print(f"\nTesting ZZ feature map with first training sample:")
sample = X_train_pca[0]
print(f"Sample features (first 4): {sample[:4]}")

# Bind parameters to create a concrete circuit for this sample
qc_zz_sample = zz_feature_map.assign_parameters(dict(zip(zz_feature_params, sample)))
print(f"\nZZ Circuit with sample features bound:")
print(f"  Circuit depth: {qc_zz_sample.depth()}")
print(f"  Number of gates: {len(qc_zz_sample.data)}")
print(f"  First few gates:")
for i, instruction in enumerate(qc_zz_sample.data[:8]):
    gate_name = instruction.operation.name
    qubits = [qc_zz_sample.find_bit(q)[0] for q in instruction.qubits]
    if instruction.operation.params:
        params = [f"{p:.4f}" for p in instruction.operation.params]
        print(f"    Gate {i+1}: {gate_name} on qubit(s) {qubits}, angle: {params}")
    else:
        print(f"    Gate {i+1}: {gate_name} on qubit(s) {qubits}")

# Visualize circuit for a few input samples (Part 2 requirement)
print("\nVisualizing ZZ circuits for a few input samples...")
for i in range(min(3, len(X_train_pca))):  # Visualize first 3 samples
    sample = X_train_pca[i]
    qc_zz_vis = zz_feature_map.assign_parameters(dict(zip(zz_feature_params, sample)))
    
    # Save circuit diagram
    try:
        fig = qc_zz_vis.draw(output='mpl', style='iqp', scale=0.8)
        plt.savefig(f'circuit_zz_feature_map_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved circuit visualization for sample {i+1}: circuit_zz_feature_map_sample_{i+1}.png")
    except Exception as e:
        # Properly handle exception
        print(f"  Skipping visualization for sample {i+1}: {e}")

# ============================================================================
# Part 3: Quantum Kernel Computation
# ============================================================================
print(f"\n{'='*60}")
print("Part 3: Quantum Kernel Computation")
print(f"{'='*60}")

# Set up the sampler and fidelity computation
# Use StatevectorSampler (V2) for exact simulation
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)

# Function to create a feature map circuit that can be used with FidelityQuantumKernel
def create_feature_map_circuit_for_kernel(base_circuit, feature_params, n_features):
    """
    Create a feature map circuit compatible with FidelityQuantumKernel.
    This wraps the parameterized circuit in a way that FidelityQuantumKernel can use.
    """
    # FidelityQuantumKernel expects a circuit that takes a single parameter vector
    # We'll create a wrapper that binds parameters correctly
    return base_circuit

# Compute quantum kernel for Basic Feature Map
print(f"\n{'='*60}")
print("Computing Quantum Kernel: Basic Feature Map")
print(f"{'='*60}")

# Create kernel for basic feature map
basic_kernel = FidelityQuantumKernel(
    feature_map=quantum_feature_map,
    fidelity=fidelity
)

print("Computing kernel matrix for training set...")
print(f"  Input shape: {X_train_pca.shape}")
print("  This may take a few moments...")
K_train_basic = basic_kernel.evaluate(X_train_pca)
print("  Kernel computation completed!")

print(f"Kernel matrix shape: {K_train_basic.shape}")
print(f"Kernel matrix min: {K_train_basic.min():.4f}, max: {K_train_basic.max():.4f}")
print(f"Kernel matrix mean: {K_train_basic.mean():.4f}")

# Visualize kernel matrix as heatmap (first 50 entries only as per requirements)
plt.figure(figsize=(10, 8))
plt.imshow(K_train_basic[:50, :50], cmap='viridis', aspect='auto')  # Show first 50×50 only
plt.colorbar(label='Kernel Value')
plt.title('Quantum Kernel Matrix - Basic Feature Map\n(First 50 Training Samples)')
plt.xlabel('Sample j')
plt.ylabel('Sample i')
plt.tight_layout()
plt.savefig('quantum_kernel_basic_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved heatmap to: quantum_kernel_basic_heatmap.png")
plt.close()

# Compute quantum kernel for ZZ Feature Map
print(f"\n{'='*60}")
print("Computing Quantum Kernel: ZZ Feature Map")
print(f"{'='*60}")

# Create kernel for ZZ feature map
zz_kernel = FidelityQuantumKernel(
    feature_map=zz_feature_map,
    fidelity=fidelity
)

print("Computing kernel matrix for training set...")
print(f"  Input shape: {X_train_pca.shape}")
print("  This may take a few moments...")
K_train_zz = zz_kernel.evaluate(X_train_pca)
print("  Kernel computation completed!")

print(f"Kernel matrix shape: {K_train_zz.shape}")
print(f"Kernel matrix min: {K_train_zz.min():.4f}, max: {K_train_zz.max():.4f}")
print(f"Kernel matrix mean: {K_train_zz.mean():.4f}")

# Visualize kernel matrix as heatmap (first 50 entries only as per requirements)
plt.figure(figsize=(10, 8))
plt.imshow(K_train_zz[:50, :50], cmap='viridis', aspect='auto')  # Show first 50×50 only
plt.colorbar(label='Kernel Value')
plt.title('Quantum Kernel Matrix - ZZ Feature Map\n(First 50 Training Samples)')
plt.xlabel('Sample j')
plt.ylabel('Sample i')
plt.tight_layout()
plt.savefig('quantum_kernel_zz_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved heatmap to: quantum_kernel_zz_heatmap.png")
plt.close()

# Comments on the kernel matrices (analyzing first 50 entries for visualization)
print(f"\n{'='*60}")
print("Kernel Matrix Analysis (First 50 Entries)")
print(f"{'='*60}")

# Analyze first 50×50 submatrix for visualization
K_basic_50 = K_train_basic[:50, :50]
K_zz_50 = K_train_zz[:50, :50]

print("\nBasic Feature Map Kernel (first 50×50):")
print(f"  - Diagonal values (self-similarity): {np.diag(K_basic_50).mean():.4f} ± {np.diag(K_basic_50).std():.4f}")
print(f"  - Off-diagonal mean: {K_basic_50[np.triu_indices_from(K_basic_50, k=1)].mean():.4f}")
print(f"  - Kernel matrix is symmetric: {np.allclose(K_basic_50, K_basic_50.T)}")

print("\nZZ Feature Map Kernel (first 50×50):")
print(f"  - Diagonal values (self-similarity): {np.diag(K_zz_50).mean():.4f} ± {np.diag(K_zz_50).std():.4f}")
print(f"  - Off-diagonal mean: {K_zz_50[np.triu_indices_from(K_zz_50, k=1)].mean():.4f}")
print(f"  - Kernel matrix is symmetric: {np.allclose(K_zz_50, K_zz_50.T)}")

print("\nComparison (first 50×50):")
print(f"  - Basic kernel diagonal mean: {np.diag(K_basic_50).mean():.4f}")
print(f"  - ZZ kernel diagonal mean: {np.diag(K_zz_50).mean():.4f}")
print(f"  - Basic kernel off-diagonal mean: {K_basic_50[np.triu_indices_from(K_basic_50, k=1)].mean():.4f}")
print(f"  - ZZ kernel off-diagonal mean: {K_zz_50[np.triu_indices_from(K_zz_50, k=1)].mean():.4f}")

print("\nInterpretation:")
print("  - Diagonal values should be close to 1.0 (perfect self-similarity)")
print("  - Off-diagonal values represent similarity between different samples")
print("  - Higher off-diagonal values indicate more similar feature representations")
print("  - The ZZ feature map with feature product interactions may create different similarity patterns")
print("  - Note: Analysis shown for first 50 entries (visualization subset)")

# Compute kernel matrices for SVM training
print(f"\n{'='*60}")
print("Computing Kernel Matrices for SVM Training")
print(f"{'='*60}")

print("\nComputing training and test kernel matrices...")
print(f"  Training samples: {len(X_train_pca)}")
print(f"  Test samples: {len(X_test_pca)}")

print("\nComputing training kernel matrix (Basic Feature Map)...")
K_train_full_basic = K_train_basic  # Already computed above
print(f"  Shape: {K_train_full_basic.shape}")

print("\nComputing test kernel matrix (Basic Feature Map)...")
K_test_basic = basic_kernel.evaluate(X_test_pca, X_train_pca)
print(f"  Shape: {K_test_basic.shape}")

print("\nComputing training kernel matrix (ZZ Feature Map)...")
K_train_full_zz = K_train_zz  # Already computed above
print(f"  Shape: {K_train_full_zz.shape}")

print("\nComputing test kernel matrix (ZZ Feature Map)...")
K_test_zz = zz_kernel.evaluate(X_test_pca, X_train_pca)
print(f"  Shape: {K_test_zz.shape}")

print("\nQuantum kernel computation completed!")
print("Kernel matrices are ready for SVM training.")

# ============================================================================
# Part 4: Classical SVM Training and Evaluation
# ============================================================================
print(f"\n{'='*60}")
print("Part 4: Classical SVM Training and Evaluation")
print(f"{'='*60}")

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Labels are already binary (converted earlier)
print(f"\nLabel encoding: 3 -> 0, 8 -> 1")
print(f"Training labels: {np.sum(y_train_binary == 0)} zeros, {np.sum(y_train_binary == 1)} ones")
print(f"Test labels: {np.sum(y_test_binary == 0)} zeros, {np.sum(y_test_binary == 1)} ones")

# Function to train and evaluate SVM
def train_and_evaluate_svm(K_train, K_test, y_train, y_test, kernel_name, param_grid=None):
    """
    Train SVM with precomputed kernel and evaluate performance.
    """
    print(f"\n{'='*60}")
    print(f"Training SVM: {kernel_name}")
    print(f"{'='*60}")
    
    # Default parameter grid for C
    if param_grid is None:
        param_grid = {'C': [0.1, 1, 10, 100, 1000]}
    
    # Grid search for hyperparameter tuning
    # Use 3-fold CV for smaller datasets, 5-fold for larger
    n_folds = 3 if len(K_train) < 100 else 5
    print(f"  Performing grid search for hyperparameter tuning ({n_folds}-fold CV)...")
    svm = SVC(kernel='precomputed')
    grid_search = GridSearchCV(svm, param_grid, cv=n_folds, scoring='accuracy', n_jobs=-1)
    grid_search.fit(K_train, y_train)
    
    best_C = grid_search.best_params_['C']
    best_score = grid_search.best_score_
    print(f"  Best C parameter: {best_C}")
    print(f"  Best CV accuracy: {best_score:.4f}")
    
    # Train final model with best parameters
    best_svm = SVC(kernel='precomputed', C=best_C)
    best_svm.fit(K_train, y_train)
    
    # Predictions
    y_train_pred = best_svm.predict(K_train)
    y_test_pred = best_svm.predict(K_test)
    
    # Evaluate
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n  Results:")
    print(f"    Training Accuracy: {train_accuracy:.4f}")
    print(f"    Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\n  Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Digit 3', 'Digit 8']))
    
    print(f"\n  Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"    True Negatives (3->3): {cm[0,0]}")
    print(f"    False Positives (3->8): {cm[0,1]}")
    print(f"    False Negatives (8->3): {cm[1,0]}")
    print(f"    True Positives (8->8): {cm[1,1]}")
    
    return {
        'model': best_svm,
        'C': best_C,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test_pred': y_test_pred
    }

# Train SVMs on Quantum Kernels
print(f"\n{'='*60}")
print("Quantum Kernel SVMs")
print(f"{'='*60}")

# Basic Feature Map SVM
results_basic = train_and_evaluate_svm(
    K_train_full_basic, K_test_basic, 
    y_train_binary, y_test_binary,
    "Basic Quantum Feature Map"
)

# ZZ Feature Map SVM
results_zz = train_and_evaluate_svm(
    K_train_full_zz, K_test_zz,
    y_train_binary, y_test_binary,
    "ZZ Quantum Feature Map"
)

# Train Classical SVMs for Comparison
print(f"\n{'='*60}")
print("Classical Kernel SVMs (for comparison)")
print(f"{'='*60}")

# RBF Kernel
print(f"\n{'='*60}")
print("Training SVM with RBF Kernel")
print(f"{'='*60}")
param_grid_rbf = {'C': [0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]}
svm_rbf = SVC(kernel='rbf')
n_folds_classical = 3 if len(X_train_pca) < 100 else 5
grid_search_rbf = GridSearchCV(svm_rbf, param_grid_rbf, cv=n_folds_classical, scoring='accuracy', n_jobs=-1)
grid_search_rbf.fit(X_train_pca, y_train_binary)
best_svm_rbf = SVC(kernel='rbf', C=grid_search_rbf.best_params_['C'], 
                   gamma=grid_search_rbf.best_params_['gamma'])
best_svm_rbf.fit(X_train_pca, y_train_binary)
y_test_pred_rbf = best_svm_rbf.predict(X_test_pca)
test_accuracy_rbf = accuracy_score(y_test_binary, y_test_pred_rbf)
print(f"  Best parameters: C={grid_search_rbf.best_params_['C']}, gamma={grid_search_rbf.best_params_['gamma']}")
print(f"  Test Accuracy: {test_accuracy_rbf:.4f}")

# Polynomial Kernel
print(f"\n{'='*60}")
print("Training SVM with Polynomial Kernel")
print(f"{'='*60}")
param_grid_poly = {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4]}
svm_poly = SVC(kernel='poly')
grid_search_poly = GridSearchCV(svm_poly, param_grid_poly, cv=n_folds_classical, scoring='accuracy', n_jobs=-1)
grid_search_poly.fit(X_train_pca, y_train_binary)
best_svm_poly = SVC(kernel='poly', C=grid_search_poly.best_params_['C'],
                    degree=grid_search_poly.best_params_['degree'])
best_svm_poly.fit(X_train_pca, y_train_binary)
y_test_pred_poly = best_svm_poly.predict(X_test_pca)
test_accuracy_poly = accuracy_score(y_test_binary, y_test_pred_poly)
print(f"  Best parameters: C={grid_search_poly.best_params_['C']}, degree={grid_search_poly.best_params_['degree']}")
print(f"  Test Accuracy: {test_accuracy_poly:.4f}")

# Linear Kernel
print(f"\n{'='*60}")
print("Training SVM with Linear Kernel")
print(f"{'='*60}")
param_grid_linear = {'C': [0.1, 1, 10, 100, 1000]}
svm_linear = SVC(kernel='linear')
grid_search_linear = GridSearchCV(svm_linear, param_grid_linear, cv=n_folds_classical, scoring='accuracy', n_jobs=-1)
grid_search_linear.fit(X_train_pca, y_train_binary)
best_svm_linear = SVC(kernel='linear', C=grid_search_linear.best_params_['C'])
best_svm_linear.fit(X_train_pca, y_train_binary)
y_test_pred_linear = best_svm_linear.predict(X_test_pca)
test_accuracy_linear = accuracy_score(y_test_binary, y_test_pred_linear)
print(f"  Best parameters: C={grid_search_linear.best_params_['C']}")
print(f"  Test Accuracy: {test_accuracy_linear:.4f}")

# Comparison Table
print(f"\n{'='*60}")
print("Performance Comparison Summary")
print(f"{'='*60}")

results_summary = {
    'Basic Quantum Feature Map': results_basic['test_accuracy'],
    'ZZ Quantum Feature Map': results_zz['test_accuracy'],
    'RBF Kernel (Classical)': test_accuracy_rbf,
    'Polynomial Kernel (Classical)': test_accuracy_poly,
    'Linear Kernel (Classical)': test_accuracy_linear
}

print("\nTest Accuracy Comparison:")
print("-" * 60)
for method, accuracy in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method:35s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Find best method
best_method = max(results_summary.items(), key=lambda x: x[1])
print(f"\nBest performing method: {best_method[0]} with {best_method[1]:.4f} ({best_method[1]*100:.2f}%) accuracy")

# Save results to file
print(f"\n{'='*60}")
print("Saving Results")
print(f"{'='*60}")

results_file = "svm_results_summary.txt"
with open(results_file, 'w') as f:
    f.write("Quantum Kernel Classification - SVM Results Summary\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Quantum Kernel SVMs:\n")
    f.write(f"  Basic Quantum Feature Map: {results_basic['test_accuracy']:.4f}\n")
    f.write(f"    Best C: {results_basic['C']}\n")
    f.write(f"  ZZ Quantum Feature Map: {results_zz['test_accuracy']:.4f}\n")
    f.write(f"    Best C: {results_zz['C']}\n\n")
    
    f.write("Classical Kernel SVMs:\n")
    f.write(f"  RBF Kernel: {test_accuracy_rbf:.4f}\n")
    f.write(f"  Polynomial Kernel: {test_accuracy_poly:.4f}\n")
    f.write(f"  Linear Kernel: {test_accuracy_linear:.4f}\n\n")
    
    f.write("Best Method: " + best_method[0] + f" ({best_method[1]:.4f})\n")

print(f"Results saved to: {results_file}")
print("\nPart 4 (SVM Training and Evaluation) completed!")

# ============================================================================
# Part 5: Angle Embedding with QSVM
# ============================================================================
print(f"\n{'='*60}")
print("Part 5: Angle Embedding with QSVM")
print(f"{'='*60}")

print("\nNote: PCA has already been applied in Part 2")
print(f"Using PCA-transformed data: {X_train_pca.shape[0]} training samples, {X_test_pca.shape[0]} test samples")
print(f"Features: {X_train_pca.shape[1]} (reduced from 64 to 16 via PCA)")
print(f"Dataset limited to {n_train_samples} training and {n_test_samples} test samples for faster computation")

# Angle Encoding Feature Map
# Angle encoding directly encodes features as rotation angles on qubits
def create_angle_encoding_feature_map(n_qubits, n_features):
    """
    Create an angle encoding feature map for QSVM.
    This uses only RY rotations to encode all features (simpler than basic map).
    This creates a meaningful difference from the basic feature map which uses
    multiple rotation types (RX, RY, RZ) while angle encoding uses only RY.
    
    For 16 features and 4 qubits: encodes all 16 features (4 per qubit using RY rotations).
    """
    # Create parameter vector for features
    features = ParameterVector('x', n_features)
    
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    # Encode all features: 4 features per qubit using RY rotations (angle encoding)
    features_per_qubit = n_features // n_qubits
    
    for qubit in range(n_qubits):
        start_idx = qubit * features_per_qubit
        # Apply RY rotations for each feature (angle encoding)
        for i in range(features_per_qubit):
            if start_idx + i < n_features:
                qc.ry(features[start_idx + i], qubit)
    
    return qc, features

# Create angle encoding feature map
print(f"\n{'='*60}")
print("Creating Angle Encoding Feature Map")
print(f"{'='*60}")

angle_feature_map, angle_feature_params = create_angle_encoding_feature_map(n_qubits, n_features)

print(f"\nAngle Encoding Feature Map:")
print(f"Number of qubits: {n_qubits}")
print(f"Number of features: {n_features}")
print(f"Features per qubit: {n_features // n_qubits}")
print(f"Circuit depth: {angle_feature_map.depth()}")
print(f"Number of gates: {len(angle_feature_map.data)}")

# Show circuit structure
print(f"\nCircuit gates (first 8):")
for i, instruction in enumerate(angle_feature_map.data[:8]):
    gate_name = instruction.operation.name
    qubits = [angle_feature_map.find_bit(q)[0] for q in instruction.qubits]
    params = [p.name if hasattr(p, 'name') else str(p) for p in instruction.operation.params]
    print(f"  Gate {i+1}: {gate_name} on qubit(s) {qubits}, params: {params}")

print("\nAngle Encoding Characteristics:")
print("  - Uses only RY rotations to encode all features as angles")
print("  - Simpler than basic feature map (only RY, no RX/RZ)")
print("  - Simpler than entangled feature maps (no CNOT gates)")
print("  - Direct angle encoding: each feature value becomes a rotation angle")
print("  - Encodes all 16 features (4 per qubit) using RY rotations")
print("  - Suitable for QSVM as it creates a quantum feature space")

# Compute quantum kernel with angle encoding
print(f"\n{'='*60}")
print("Computing Quantum Kernel with Angle Encoding")
print(f"{'='*60}")

# Create kernel for angle encoding feature map
angle_kernel = FidelityQuantumKernel(
    feature_map=angle_feature_map,
    fidelity=fidelity
)

# Compute kernel matrices for QSVM training
print(f"\n{'='*60}")
print("Computing Kernel Matrices for QSVM Training")
print(f"{'='*60}")

print("\nComputing training kernel matrix (Angle Encoding)...")
print(f"  Input shape: {X_train_pca.shape}")
print("  This may take a few moments...")
K_train_angle = angle_kernel.evaluate(X_train_pca)
print("  Kernel computation completed!")
print(f"  Shape: {K_train_angle.shape}")
print(f"  Min: {K_train_angle.min():.4f}, Max: {K_train_angle.max():.4f}, Mean: {K_train_angle.mean():.4f}")

print("\nComputing test kernel matrix (Angle Encoding)...")
K_test_angle = angle_kernel.evaluate(X_test_pca, X_train_pca)
print(f"  Shape: {K_test_angle.shape}")

# Train QSVM using angle encoding kernel
print(f"\n{'='*60}")
print("Training QSVM with Angle Encoding")
print(f"{'='*60}")

print("\nQSVM = Quantum Support Vector Machine")
print("Implementation: Classical SVM with quantum kernel (angle encoding)")

# Train QSVM with angle encoding kernel
results_angle = train_and_evaluate_svm(
    K_train_angle, K_test_angle,
    y_train_binary, y_test_binary,
    "QSVM with Angle Encoding"
)

# Compare to Part 4 results
print(f"\n{'='*60}")
print("Comparison: Part 4 vs Part 5 (Angle Encoding)")
print(f"{'='*60}")

comparison_results = {
    'Part 4 - Basic Quantum Feature Map': results_basic['test_accuracy'],
    'Part 4 - ZZ Quantum Feature Map': results_zz['test_accuracy'],
    'Part 4 - RBF Kernel (Classical)': test_accuracy_rbf,
    'Part 4 - Polynomial Kernel (Classical)': test_accuracy_poly,
    'Part 4 - Linear Kernel (Classical)': test_accuracy_linear,
    'Part 5 - QSVM with Angle Encoding': results_angle['test_accuracy']
}

print("\nTest Accuracy Comparison:")
print("-" * 70)
for method, accuracy in sorted(comparison_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method:45s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Analysis
print(f"\n{'='*60}")
print("Analysis: How Angle Encoding Differs from Part 4")
print(f"{'='*60}")

print("\n1. Feature Map Comparison:")
print("   - Basic Feature Map (Part 4): Uses RX, RY, RZ rotations, encodes all 16 features")
print("   - ZZ Feature Map (Part 4): Uses Hadamard, RZ rotations, ZZ interactions with feature products")
print("   - Angle Encoding (Part 5): Uses only RY rotations, encodes all 16 features (simpler than basic)")

print("\n2. Circuit Complexity:")
print(f"   - Basic Feature Map: {quantum_feature_map.depth()} depth, {len(quantum_feature_map.data)} gates")
print(f"   - ZZ Feature Map: {zz_feature_map.depth()} depth, {len(zz_feature_map.data)} gates")
print(f"   - Angle Encoding: {angle_feature_map.depth()} depth, {len(angle_feature_map.data)} gates")

print("\n3. Performance Comparison:")
angle_vs_basic = results_angle['test_accuracy'] - results_basic['test_accuracy']
angle_vs_zz = results_angle['test_accuracy'] - results_zz['test_accuracy']
angle_vs_rbf = results_angle['test_accuracy'] - test_accuracy_rbf

print(f"   - Angle Encoding vs Basic: {angle_vs_basic:+.4f} ({angle_vs_basic*100:+.2f}%)")
print(f"   - Angle Encoding vs ZZ: {angle_vs_zz:+.4f} ({angle_vs_zz*100:+.2f}%)")
print(f"   - Angle Encoding vs RBF: {angle_vs_rbf:+.4f} ({angle_vs_rbf*100:+.2f}%)")

if results_angle['test_accuracy'] > max(results_basic['test_accuracy'], results_zz['test_accuracy']):
    print("\n   -> Angle encoding performs better than quantum feature maps from Part 4")
elif results_angle['test_accuracy'] < min(results_basic['test_accuracy'], results_zz['test_accuracy']):
    print("\n   -> Angle encoding performs worse than quantum feature maps from Part 4")
else:
    print("\n   -> Angle encoding performs similarly to quantum feature maps from Part 4")

print("\n4. Key Differences:")
print("   - Angle encoding is simpler (no entanglement)")
print("   - Direct feature-to-angle mapping")
print("   - May be more interpretable but potentially less expressive")
print("   - Faster to compute due to simpler circuit")

# Update results file
print(f"\n{'='*60}")
print("Updating Results File")
print(f"{'='*60}")

with open(results_file, 'a') as f:
    f.write("\n" + "="*60 + "\n")
    f.write("Part 5: Angle Embedding with QSVM\n")
    f.write("="*60 + "\n\n")
    f.write(f"QSVM with Angle Encoding: {results_angle['test_accuracy']:.4f}\n")
    f.write(f"  Best C: {results_angle['C']}\n\n")
    f.write("Comparison to Part 4:\n")
    for method, accuracy in sorted(comparison_results.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  {method}: {accuracy:.4f}\n")

print(f"Results updated in: {results_file}")
print("\nPart 5 (Angle Embedding with QSVM) completed!")
print(f"\n{'='*60}")
print("ALL PARTS COMPLETED!")
print(f"{'='*60}")
print("\nSummary of completed parts:")
print("  [OK] Part 1: Theoretical Background (documentation needed)")
print("  [OK] Part 2: Quantum Feature Maps (Basic + ZZ)")
print("  [OK] Part 3: Quantum Kernel Computation")
print("  [OK] Part 4: Classical SVM Training and Evaluation")
print("  [OK] Part 5: Angle Embedding with QSVM")

