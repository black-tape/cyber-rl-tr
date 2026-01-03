"""
Minimum Viable Experiment (MVE) for Validating Core Hypothesis

This script runs a focused experiment to validate whether our multi-agent
approach provides meaningful improvements over baselines.

Success Criteria (Must achieve to proceed):
- F1 Score improvement >= 5% over FedAvg baseline
- Convergence speedup >= 20% (fewer rounds to target accuracy)
- Robustness to malicious clients (degradation < baselines)

Experimental Setup:
- Dataset: NSL-KDD (easy to download and process)
- Clients: 5 (3 benign, 2 malicious = 40% attack)
- Non-IID: Dirichlet alpha = 0.1 (extreme heterogeneity)
- Attack: Label Flipping (50% of labels flipped)
- Model: Simple MLP (avoid model capacity hiding differences)
- Rounds: 50 (enough to see convergence)

Comparison Methods:
1. FedAvg (uniform weights - baseline)
2. Krum (robust aggregation - baseline)
3. Trimmed Mean (robust aggregation - baseline)
4. Ours (Multi-Agent with anomaly detection + trust + adaptive weights)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.dataset_loader import NSLKDDProcessor
from data.non_iid import DirichletPartitioner
from attack.poisoning import AttackCoordinator
from baseline.robust_aggregation import get_aggregator
from agent.simple_agents import MultiAgentFederatedSystem


# Simple MLP model for binary classification
class SimpleMLP(nn.Module):
    """Lightweight MLP to avoid model capacity hiding differences."""

    def __init__(self, input_dim: int = 122, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FederatedTrainer:
    """Handles federated training for different aggregation methods."""

    def __init__(
        self,
        model_class,
        input_dim: int,
        hidden_dims: List[int],
        lr: float = 0.01,
        device: str = 'cpu'
    ):
        self.model_class = model_class
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.device = device

        # Initialize global model
        self.global_model = model_class(input_dim, hidden_dims).to(device)
        self.criterion = nn.BCELoss()

    def train_client(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5
    ) -> Tuple[OrderedDict, float]:
        """
        Train a client locally.

        Returns:
            (model_state_dict, loss): Client's trained model and final loss
        """
        # Create local model (copy of global)
        local_model = self.model_class(self.input_dim, self.hidden_dims).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())

        # Convert data to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        # Train
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
        local_model.train()

        total_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / epochs

        return local_model.state_dict(), avg_loss

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate global model on test set.

        Returns:
            Dictionary with metrics
        """
        self.global_model.eval()

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test)

        with torch.no_grad():
            outputs = self.global_model(X_tensor).cpu().numpy()
            predictions = (outputs > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def update_global_model(self, new_state_dict: OrderedDict):
        """Update global model with aggregated weights."""
        self.global_model.load_state_dict(new_state_dict)


def run_mve_experiment(
    method_name: str,
    aggregator,
    client_datasets: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_coordinator: AttackCoordinator,
    num_rounds: int = 50,
    local_epochs: int = 5,
    input_dim: int = 122
):
    """
    Run a single MVE experiment with a specific aggregation method.

    Args:
        method_name: Name of the method (for logging)
        aggregator: Aggregation method instance
        client_datasets: List of (X, y) tuples for each client
        X_test, y_test: Test data
        attack_coordinator: Coordinates attacks on malicious clients
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per round
        input_dim: Input feature dimension

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {method_name}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = FederatedTrainer(SimpleMLP, input_dim, [64, 32], device=device)

    # Track metrics
    history = {
        'rounds': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    start_time = time.time()

    # Federated training loop
    for round_idx in range(num_rounds):
        print(f"\n[Round {round_idx + 1}/{num_rounds}]")

        client_updates = []
        client_performances = []

        # Train each client
        for client_id, (X_client, y_client) in enumerate(client_datasets):
            # Apply data poisoning if malicious
            X_train, y_train = attack_coordinator.apply_data_attack(
                client_id, X_client, y_client
            )

            # Train locally
            model_update, loss = trainer.train_client(X_train, y_train, local_epochs)

            # Apply model poisoning if malicious (for model-level attacks)
            model_update = attack_coordinator.apply_model_attack(client_id, model_update)

            # Evaluate client performance (on clean local test set)
            # Use a small portion of client data as validation
            val_size = min(100, len(X_client) // 5)
            X_val = X_client[:val_size]
            y_val = y_client[:val_size]

            trainer.global_model.load_state_dict(model_update)
            perf = trainer.evaluate(X_val, y_val)

            client_updates.append(model_update)
            client_performances.append(perf['f1'])  # Use F1 as performance metric

            malicious_flag = "[MALICIOUS]" if attack_coordinator.is_malicious(client_id) else ""
            print(f"  Client {client_id} {malicious_flag}: loss={loss:.4f}, F1={perf['f1']:.4f}")

        # Aggregate updates
        if isinstance(aggregator, MultiAgentFederatedSystem):
            # Our multi-agent method
            global_update = aggregator.aggregate_updates(
                client_updates,
                trainer.global_model.state_dict(),
                client_performances
            )
        else:
            # Baseline methods
            if hasattr(aggregator, 'aggregate'):
                global_update = aggregator.aggregate(client_updates)
            else:
                # FedAvg (uniform)
                global_update = aggregator.aggregate(
                    client_updates,
                    client_weights=[1.0 / len(client_updates)] * len(client_updates)
                )

        # Update global model
        trainer.update_global_model(global_update)

        # Evaluate on test set
        metrics = trainer.evaluate(X_test, y_test)

        history['rounds'].append(round_idx + 1)
        history['accuracy'].append(metrics['accuracy'])
        history['f1'].append(metrics['f1'])
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])

        print(f"  Global: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

    training_time = time.time() - start_time

    # Compute summary statistics
    final_metrics = {
        'method': method_name,
        'final_accuracy': history['accuracy'][-1],
        'final_f1': history['f1'][-1],
        'final_precision': history['precision'][-1],
        'final_recall': history['recall'][-1],
        'best_f1': max(history['f1']),
        'rounds_to_90pct_f1': None,
        'training_time': training_time,
        'history': history
    }

    # Find convergence round (90% of best F1)
    target_f1 = 0.9 * final_metrics['best_f1']
    for i, f1 in enumerate(history['f1']):
        if f1 >= target_f1:
            final_metrics['rounds_to_90pct_f1'] = i + 1
            break

    return final_metrics


def plot_results(all_results: List[Dict]):
    """Plot comparison of all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['accuracy', 'f1', 'precision', 'recall']
    titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        for result in all_results:
            ax.plot(
                result['history']['rounds'],
                result['history'][metric],
                label=result['method'],
                linewidth=2
            )

        ax.set_xlabel('Federated Round')
        ax.set_ylabel(title)
        ax.set_title(f'{title} over Rounds')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mve_results.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot saved to mve_results.png")


def print_summary_table(all_results: List[Dict]):
    """Print summary table comparing all methods."""
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Method':<20} {'Final F1':<12} {'Best F1':<12} {'Rounds to 90%':<15} {'Time (s)':<10}")
    print(f"{'-'*80}")

    baseline_f1 = None
    for result in all_results:
        if result['method'] == 'FedAvg (Baseline)':
            baseline_f1 = result['final_f1']

        rounds_str = str(result['rounds_to_90pct_f1']) if result['rounds_to_90pct_f1'] else 'N/A'

        print(f"{result['method']:<20} {result['final_f1']:<12.4f} {result['best_f1']:<12.4f} "
              f"{rounds_str:<15} {result['training_time']:<10.2f}")

    # Compute improvements
    if baseline_f1:
        print(f"\n{'='*80}")
        print("IMPROVEMENT OVER BASELINE (FedAvg)")
        print(f"{'='*80}")

        for result in all_results:
            if result['method'] != 'FedAvg (Baseline)':
                f1_improvement = (result['final_f1'] - baseline_f1) / baseline_f1 * 100
                print(f"{result['method']:<20} F1 Improvement: {f1_improvement:>+6.2f}%")


def main():
    """Run the Minimum Viable Experiment."""
    print("="*80)
    print("MINIMUM VIABLE EXPERIMENT (MVE)")
    print("="*80)

    # Configuration
    NUM_CLIENTS = 5
    MALICIOUS_RATIO = 0.4  # 40% malicious (2 out of 5)
    DIRICHLET_ALPHA = 0.1  # Extreme Non-IID
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5

    # Load and preprocess NSL-KDD
    print("\n[1] Loading NSL-KDD dataset...")
    processor = NSLKDDProcessor()
    try:
        X_train_full, y_train_full, X_test, y_test = processor.preprocess()
        print(f"  Train samples: {len(X_train_full)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train_full.shape[1]}")
    except Exception as e:
        print(f"  Error loading NSL-KDD: {e}")
        print("  Using synthetic data for testing...")
        # Generate synthetic data
        X_train_full = np.random.randn(5000, 122)
        y_train_full = np.random.randint(0, 2, 5000)
        X_test = np.random.randn(1000, 122)
        y_test = np.random.randint(0, 2, 1000)

    input_dim = X_train_full.shape[1]

    # Partition data with Dirichlet (extreme Non-IID)
    print(f"\n[2] Partitioning data (Dirichlet alpha={DIRICHLET_ALPHA})...")
    partitioner = DirichletPartitioner(alpha=DIRICHLET_ALPHA, min_samples_per_client=100)
    client_datasets = partitioner.partition(X_train_full, y_train_full, NUM_CLIENTS)

    for i, (X, y) in enumerate(client_datasets):
        print(f"  Client {i}: {len(X)} samples, Attack ratio: {y.mean():.2%}")

    # Setup attack coordinator
    print(f"\n[3] Setting up attacks ({MALICIOUS_RATIO:.0%} malicious clients)...")
    attack_coordinator = AttackCoordinator(
        num_clients=NUM_CLIENTS,
        malicious_ratio=MALICIOUS_RATIO,
        attack_type='label_flipping',
        flip_ratio=0.5  # Flip 50% of labels
    )
    print(f"  Malicious clients: {sorted(attack_coordinator.malicious_clients)}")

    # Prepare aggregators
    aggregators = [
        ('FedAvg (Baseline)', get_aggregator('fedavg')),
        ('Krum', get_aggregator('krum', num_byzantine=2)),
        ('Trimmed Mean', get_aggregator('trimmed_mean', trim_ratio=0.2)),
        ('Ours (Multi-Agent)', MultiAgentFederatedSystem(
            num_clients=NUM_CLIENTS,
            trust_alpha=0.7,
            temperature=1.0
        ))
    ]

    # Run experiments
    print(f"\n[4] Running experiments ({NUM_ROUNDS} rounds)...")
    all_results = []

    for method_name, aggregator in aggregators:
        result = run_mve_experiment(
            method_name=method_name,
            aggregator=aggregator,
            client_datasets=client_datasets,
            X_test=X_test,
            y_test=y_test,
            attack_coordinator=attack_coordinator,
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            input_dim=input_dim
        )
        all_results.append(result)

    # Display results
    print_summary_table(all_results)

    # Plot results
    plot_results(all_results)

    # Check success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}")

    baseline_result = next(r for r in all_results if r['method'] == 'FedAvg (Baseline)')
    ours_result = next(r for r in all_results if r['method'] == 'Ours (Multi-Agent)')

    f1_improvement = (ours_result['final_f1'] - baseline_result['final_f1']) / baseline_result['final_f1'] * 100

    print(f"\nCriterion 1: F1 improvement >= 5%")
    print(f"  Result: {f1_improvement:+.2f}%  {'[PASS]' if f1_improvement >= 5 else '[FAIL]'}")

    if baseline_result['rounds_to_90pct_f1'] and ours_result['rounds_to_90pct_f1']:
        convergence_speedup = (
            (baseline_result['rounds_to_90pct_f1'] - ours_result['rounds_to_90pct_f1']) /
            baseline_result['rounds_to_90pct_f1'] * 100
        )
        print(f"\nCriterion 2: Convergence speedup >= 20%")
        print(f"  Result: {convergence_speedup:+.2f}%  {'[PASS]' if convergence_speedup >= 20 else '[FAIL]'}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
