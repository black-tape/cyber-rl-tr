"""
Non-IID Data Partitioning for Federated Learning

Implements various strategies to partition data in non-IID (non-independent
and identically distributed) ways:
- Dirichlet Distribution: Control degree of heterogeneity with alpha parameter
- Label Skew: Each client has different label distribution
- Quantity Skew: Clients have different amounts of data
- Feature Skew: Different feature distributions (domain shift)

These simulate realistic federated scenarios where clients have different data.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class DirichletPartitioner:
    """
    Dirichlet Distribution Partitioning.

    Uses Dirichlet distribution to control the degree of label heterogeneity.

    Alpha parameter controls heterogeneity:
    - alpha -> 0: Extremely non-IID (each client has few classes)
    - alpha = 0.1: Highly non-IID (realistic challenging scenario)
    - alpha = 0.5: Moderately non-IID
    - alpha = 1.0: Mildly non-IID
    - alpha -> infinity: Approaches IID

    Reference: Hsu et al. "Measuring the effects of non-identical data
               distribution for federated visual classification" (2019)
    """

    def __init__(self, alpha: float = 0.5, min_samples_per_client: int = 10):
        """
        Args:
            alpha: Concentration parameter (lower = more heterogeneous)
            min_samples_per_client: Minimum samples each client must have
        """
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data using Dirichlet distribution.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            num_clients: Number of clients

        Returns:
            List of (X_client, y_client) tuples for each client
        """
        num_classes = len(np.unique(y))
        num_samples = len(y)

        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        # Initialize client data indices
        client_indices = [[] for _ in range(num_clients)]

        # For each class, distribute samples to clients using Dirichlet
        for class_label, indices in class_indices.items():
            np.random.shuffle(indices)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(
                [self.alpha] * num_clients
            )

            # Ensure minimum samples per client
            proportions = self._ensure_minimum_samples(
                proportions,
                len(indices),
                num_clients
            )

            # Distribute indices according to proportions
            proportions = proportions / proportions.sum()  # Normalize
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            class_splits = np.split(indices, split_points)

            # Assign to clients
            for client_id, client_split in enumerate(class_splits):
                client_indices[client_id].extend(client_split)

        # Create client datasets
        client_datasets = []
        for indices in client_indices:
            if len(indices) < self.min_samples_per_client:
                # If too few samples, redistribute
                print(f"Warning: Client has only {len(indices)} samples")

            indices = np.array(indices)
            np.random.shuffle(indices)

            X_client = X[indices]
            y_client = y[indices]
            client_datasets.append((X_client, y_client))

        return client_datasets

    def _ensure_minimum_samples(
        self,
        proportions: np.ndarray,
        total_samples: int,
        num_clients: int
    ) -> np.ndarray:
        """
        Adjust proportions to ensure minimum samples per client.

        Args:
            proportions: Initial proportions from Dirichlet
            total_samples: Total samples to distribute
            num_clients: Number of clients

        Returns:
            Adjusted proportions
        """
        min_proportion = self.min_samples_per_client / total_samples

        # Ensure no client gets too few samples
        proportions = np.maximum(proportions, min_proportion)

        return proportions


class LabelSkewPartitioner:
    """
    Label Skew Partitioning (Pathological Non-IID).

    Each client receives data from only a limited number of classes.

    Example: With 10 classes and shards_per_client=2, each client gets
             data from only 2 out of 10 classes.

    This creates extreme heterogeneity.

    Reference: McMahan et al. "Communication-efficient learning of deep
               networks from decentralized data" (AISTATS 2017)
    """

    def __init__(self, shards_per_client: int = 2):
        """
        Args:
            shards_per_client: Number of class shards each client receives
        """
        self.shards_per_client = shards_per_client

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data by label skew.

        Args:
            X: Features
            y: Labels
            num_clients: Number of clients

        Returns:
            List of (X_client, y_client) tuples
        """
        num_classes = len(np.unique(y))
        num_shards = num_clients * self.shards_per_client

        # Sort by label
        sorted_indices = np.argsort(y)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        # Divide into shards
        shard_size = len(y) // num_shards
        shard_indices = []

        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < num_shards - 1 else len(y)
            shard_indices.append((start_idx, end_idx))

        # Randomly assign shards to clients
        shard_assignments = list(range(num_shards))
        np.random.shuffle(shard_assignments)

        # Collect data for each client
        client_datasets = []
        for client_id in range(num_clients):
            # Get shards for this client
            client_shard_ids = shard_assignments[
                client_id * self.shards_per_client:
                (client_id + 1) * self.shards_per_client
            ]

            # Collect indices
            indices = []
            for shard_id in client_shard_ids:
                start_idx, end_idx = shard_indices[shard_id]
                indices.extend(range(start_idx, end_idx))

            # Create dataset
            X_client = X_sorted[indices]
            y_client = y_sorted[indices]
            client_datasets.append((X_client, y_client))

        return client_datasets


class QuantitySkewPartitioner:
    """
    Quantity Skew Partitioning.

    Clients receive different amounts of data following a power law or
    exponential distribution.

    Simulates scenarios where some clients (e.g., large organizations) have
    much more data than others (e.g., individual users).
    """

    def __init__(self, imbalance_factor: float = 0.5):
        """
        Args:
            imbalance_factor: Controls degree of imbalance (0 = balanced, 1 = very imbalanced)
                             Uses power law with exponent = -imbalance_factor
        """
        assert 0 <= imbalance_factor <= 1, "imbalance_factor must be in [0, 1]"
        self.imbalance_factor = imbalance_factor

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with quantity skew.

        Args:
            X: Features
            y: Labels
            num_clients: Number of clients

        Returns:
            List of (X_client, y_client) tuples
        """
        num_samples = len(y)

        # Generate proportions following power law
        if self.imbalance_factor > 0:
            # Power law: p_i ~ i^(-imbalance_factor)
            ranks = np.arange(1, num_clients + 1)
            proportions = ranks ** (-self.imbalance_factor)
        else:
            # Balanced
            proportions = np.ones(num_clients)

        # Normalize
        proportions = proportions / proportions.sum()

        # Compute number of samples per client
        client_sizes = (proportions * num_samples).astype(int)

        # Adjust to ensure total matches (due to rounding)
        diff = num_samples - client_sizes.sum()
        client_sizes[-1] += diff

        # Shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split data
        client_datasets = []
        start_idx = 0

        for size in client_sizes:
            end_idx = start_idx + size
            client_indices = indices[start_idx:end_idx]

            X_client = X[client_indices]
            y_client = y[client_indices]
            client_datasets.append((X_client, y_client))

            start_idx = end_idx

        return client_datasets


class CombinedPartitioner:
    """
    Combined Partitioner: Mix label skew and quantity skew.

    This creates the most realistic and challenging Non-IID scenario.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        imbalance_factor: float = 0.3,
        min_samples_per_client: int = 10
    ):
        """
        Args:
            alpha: Dirichlet alpha for label heterogeneity
            imbalance_factor: Quantity imbalance (0 = balanced)
            min_samples_per_client: Minimum samples per client
        """
        self.alpha = alpha
        self.imbalance_factor = imbalance_factor
        self.min_samples_per_client = min_samples_per_client

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition with both label and quantity skew.

        Args:
            X: Features
            y: Labels
            num_clients: Number of clients

        Returns:
            List of (X_client, y_client) tuples
        """
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # Step 1: Determine quantity distribution (quantity skew)
        if self.imbalance_factor > 0:
            ranks = np.arange(1, num_clients + 1)
            proportions = ranks ** (-self.imbalance_factor)
        else:
            proportions = np.ones(num_clients)

        proportions = proportions / proportions.sum()
        client_sizes = (proportions * num_samples).astype(int)

        # Adjust for rounding
        diff = num_samples - client_sizes.sum()
        client_sizes[-1] += diff

        # Step 2: Dirichlet partitioning for label distribution
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        # Allocate samples to clients
        client_indices = [[] for _ in range(num_clients)]

        for class_label, indices in class_indices.items():
            np.random.shuffle(indices)

            # Sample proportions from Dirichlet
            dir_proportions = np.random.dirichlet([self.alpha] * num_clients)

            # Scale by client sizes (quantity skew)
            weighted_proportions = dir_proportions * client_sizes
            weighted_proportions = weighted_proportions / weighted_proportions.sum()

            # Split class samples
            split_points = (np.cumsum(weighted_proportions) * len(indices)).astype(int)[:-1]
            class_splits = np.split(indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split)

        # Create datasets
        client_datasets = []
        for indices in client_indices:
            indices = np.array(indices)
            np.random.shuffle(indices)

            X_client = X[indices]
            y_client = y[indices]
            client_datasets.append((X_client, y_client))

        return client_datasets


# Utility functions
def analyze_partition(
    client_datasets: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict:
    """
    Analyze the heterogeneity of a data partition.

    Args:
        client_datasets: List of (X, y) tuples for each client

    Returns:
        Dictionary with statistics
    """
    num_clients = len(client_datasets)
    num_classes = len(np.unique(np.concatenate([y for _, y in client_datasets])))

    stats = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'client_sizes': [],
        'label_distributions': [],
        'classes_per_client': []
    }

    for X, y in client_datasets:
        stats['client_sizes'].append(len(y))

        # Label distribution
        unique, counts = np.unique(y, return_counts=True)
        label_dist = np.zeros(num_classes)
        label_dist[unique] = counts / len(y)
        stats['label_distributions'].append(label_dist)

        stats['classes_per_client'].append(len(unique))

    # Compute heterogeneity metrics
    stats['size_std'] = np.std(stats['client_sizes'])
    stats['size_cv'] = stats['size_std'] / np.mean(stats['client_sizes'])
    stats['avg_classes_per_client'] = np.mean(stats['classes_per_client'])

    return stats


def print_partition_stats(stats: Dict):
    """Print partition statistics in a readable format."""
    print(f"Partition Statistics:")
    print(f"  Number of clients: {stats['num_clients']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Client sizes: {stats['client_sizes']}")
    print(f"  Size std dev: {stats['size_std']:.2f}")
    print(f"  Size CV: {stats['size_cv']:.2f}")
    print(f"  Avg classes per client: {stats['avg_classes_per_client']:.2f}")
    print(f"  Classes per client: {stats['classes_per_client']}")


# Quick test
if __name__ == "__main__":
    print("Testing Non-IID Partitioning...")

    # Generate synthetic data
    np.random.seed(42)
    num_samples = 1000
    num_features = 10
    num_classes = 5
    num_clients = 5

    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)

    # Test Dirichlet
    print("\n[1] Dirichlet Partitioning (alpha=0.1):")
    partitioner = DirichletPartitioner(alpha=0.1)
    datasets = partitioner.partition(X, y, num_clients)
    stats = analyze_partition(datasets)
    print_partition_stats(stats)

    # Test Label Skew
    print("\n[2] Label Skew (2 shards per client):")
    partitioner = LabelSkewPartitioner(shards_per_client=2)
    datasets = partitioner.partition(X, y, num_clients)
    stats = analyze_partition(datasets)
    print_partition_stats(stats)

    # Test Combined
    print("\n[3] Combined (alpha=0.3, imbalance=0.5):")
    partitioner = CombinedPartitioner(alpha=0.3, imbalance_factor=0.5)
    datasets = partitioner.partition(X, y, num_clients)
    stats = analyze_partition(datasets)
    print_partition_stats(stats)

    print("\n[OK] All partitioning tests passed!")
