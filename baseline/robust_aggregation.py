"""
Robust Aggregation Baselines for Federated Learning

Implements robust aggregation methods that are resistant to Byzantine attacks:
- Krum: Select updates closest to majority
- Trimmed Mean: Remove extreme values before averaging
- Median: Coordinate-wise median aggregation
- FedProx: Proximal term regularization

These serve as baselines to compare against our multi-agent approach.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import OrderedDict


class KrumAggregator:
    """
    Krum Aggregation: Selects the update closest to the majority of updates.

    Algorithm:
    1. For each client update, compute distance to all other updates
    2. For each client, sum the distances to its k nearest neighbors
    3. Select the client with minimum sum (most "central" update)
    4. Use only that client's update as the global model

    Robust to up to (n - k - 2) Byzantine clients.

    Reference: Blanchard et al. "Machine learning with adversaries: Byzantine
               tolerant gradient descent" (NIPS 2017)
    """

    def __init__(self, num_byzantine: int = 0):
        """
        Args:
            num_byzantine: Expected number of Byzantine (malicious) clients
        """
        self.num_byzantine = num_byzantine

    def aggregate(self, client_updates: List[OrderedDict]) -> OrderedDict:
        """
        Aggregate client updates using Krum selection.

        Args:
            client_updates: List of client model state dicts

        Returns:
            Selected client's model (the one closest to majority)
        """
        num_clients = len(client_updates)

        # Number of neighbors to consider (n - f - 2 where f is Byzantine count)
        k = num_clients - self.num_byzantine - 2
        k = max(1, k)  # At least 1 neighbor

        # Flatten all updates to vectors for distance computation
        flattened_updates = []
        for update in client_updates:
            flat = self._flatten_params(update)
            flattened_updates.append(flat)

        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(flattened_updates[i] - flattened_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, sum distances to k nearest neighbors
        scores = []
        for i in range(num_clients):
            # Get distances to all other clients
            dists_to_others = distances[i].copy()
            dists_to_others[i] = np.inf  # Exclude self

            # Sum k smallest distances
            k_nearest_dists = np.partition(dists_to_others, k)[:k]
            score = np.sum(k_nearest_dists)
            scores.append(score)

        # Select client with minimum score (most central)
        selected_idx = np.argmin(scores)

        return client_updates[selected_idx]

    def _flatten_params(self, state_dict: OrderedDict) -> np.ndarray:
        """Flatten model parameters to a single vector."""
        params = []
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                params.append(param.cpu().detach().numpy().flatten())
            else:
                params.append(np.array(param).flatten())
        return np.concatenate(params)


class TrimmedMeanAggregator:
    """
    Trimmed Mean Aggregation: Remove extreme values before averaging.

    Algorithm:
    1. For each parameter, sort values across all clients
    2. Remove top beta and bottom beta fraction of values
    3. Average the remaining values

    Robust to up to beta fraction of Byzantine clients.

    Reference: Yin et al. "Byzantine-robust distributed learning: Towards
               optimal statistical rates" (ICML 2018)
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction of extreme values to remove (e.g., 0.1 = 10%)
        """
        assert 0 <= trim_ratio < 0.5, "trim_ratio must be in [0, 0.5)"
        self.trim_ratio = trim_ratio

    def aggregate(self, client_updates: List[OrderedDict]) -> OrderedDict:
        """
        Aggregate client updates using trimmed mean.

        Args:
            client_updates: List of client model state dicts

        Returns:
            Aggregated model with trimmed mean
        """
        num_clients = len(client_updates)
        num_trim = int(num_clients * self.trim_ratio)

        # Initialize result
        global_model = OrderedDict()

        # Get all parameter names
        param_names = list(client_updates[0].keys())

        # For each parameter
        for param_name in param_names:
            # Stack parameters from all clients
            param_stack = []
            for update in client_updates:
                param = update[param_name]
                if isinstance(param, torch.Tensor):
                    param_stack.append(param.cpu().detach().numpy())
                else:
                    param_stack.append(np.array(param))

            param_stack = np.array(param_stack)  # Shape: (num_clients, ...)

            # Compute trimmed mean along client dimension
            trimmed_param = self._trimmed_mean(param_stack, num_trim)

            # Convert back to tensor if needed
            if isinstance(client_updates[0][param_name], torch.Tensor):
                device = client_updates[0][param_name].device
                global_model[param_name] = torch.tensor(
                    trimmed_param, dtype=client_updates[0][param_name].dtype
                ).to(device)
            else:
                global_model[param_name] = trimmed_param

        return global_model

    def _trimmed_mean(self, values: np.ndarray, num_trim: int) -> np.ndarray:
        """
        Compute trimmed mean by removing extreme values.

        Args:
            values: Array of shape (num_clients, ...)
            num_trim: Number of extreme values to remove from each side

        Returns:
            Trimmed mean array of shape (...)
        """
        if num_trim == 0:
            return np.mean(values, axis=0)

        # Sort along client dimension
        sorted_values = np.sort(values, axis=0)

        # Remove top and bottom num_trim values
        trimmed_values = sorted_values[num_trim:-num_trim]

        # Compute mean
        return np.mean(trimmed_values, axis=0)


class MedianAggregator:
    """
    Coordinate-wise Median Aggregation.

    Algorithm:
    1. For each parameter coordinate, compute median across all clients
    2. Use median value as the aggregated parameter

    Robust to up to 50% Byzantine clients.
    Most robust but may converge slower than mean-based methods.
    """

    def aggregate(self, client_updates: List[OrderedDict]) -> OrderedDict:
        """
        Aggregate client updates using coordinate-wise median.

        Args:
            client_updates: List of client model state dicts

        Returns:
            Aggregated model with median values
        """
        # Initialize result
        global_model = OrderedDict()

        # Get all parameter names
        param_names = list(client_updates[0].keys())

        # For each parameter
        for param_name in param_names:
            # Stack parameters from all clients
            param_stack = []
            for update in client_updates:
                param = update[param_name]
                if isinstance(param, torch.Tensor):
                    param_stack.append(param.cpu().detach().numpy())
                else:
                    param_stack.append(np.array(param))

            param_stack = np.array(param_stack)  # Shape: (num_clients, ...)

            # Compute median along client dimension
            median_param = np.median(param_stack, axis=0)

            # Convert back to tensor if needed
            if isinstance(client_updates[0][param_name], torch.Tensor):
                device = client_updates[0][param_name].device
                global_model[param_name] = torch.tensor(
                    median_param, dtype=client_updates[0][param_name].dtype
                ).to(device)
            else:
                global_model[param_name] = median_param

        return global_model


class FedProxAggregator:
    """
    FedProx: Federated learning with proximal term.

    This is not a Byzantine-robust aggregation method, but a regularization
    approach that helps with Non-IID data and system heterogeneity.

    The proximal term is added during local training, not aggregation.
    This class provides the standard weighted averaging for comparison.

    Reference: Li et al. "Federated optimization in heterogeneous networks"
               (MLSys 2020)
    """

    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal term coefficient (used during client training)
        """
        self.mu = mu

    def aggregate(
        self,
        client_updates: List[OrderedDict],
        client_weights: Optional[List[float]] = None
    ) -> OrderedDict:
        """
        Aggregate using weighted average (standard FedAvg).

        Args:
            client_updates: List of client model state dicts
            client_weights: Optional weights for each client (defaults to uniform)

        Returns:
            Weighted average of client models
        """
        num_clients = len(client_updates)

        # Default to uniform weights
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients

        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # Initialize result
        global_model = OrderedDict()

        # Get all parameter names
        param_names = list(client_updates[0].keys())

        # For each parameter
        for param_name in param_names:
            # Weighted sum
            weighted_sum = None

            for i, update in enumerate(client_updates):
                param = update[param_name]
                weight = client_weights[i]

                if weighted_sum is None:
                    if isinstance(param, torch.Tensor):
                        weighted_sum = param.clone().detach() * weight
                    else:
                        weighted_sum = np.array(param) * weight
                else:
                    if isinstance(param, torch.Tensor):
                        weighted_sum += param.detach() * weight
                    else:
                        weighted_sum += np.array(param) * weight

            global_model[param_name] = weighted_sum

        return global_model


# Utility function for easy baseline selection
def get_aggregator(method: str, **kwargs):
    """
    Factory function to get aggregator by name.

    Args:
        method: One of ['krum', 'trimmed_mean', 'median', 'fedprox', 'fedavg']
        **kwargs: Additional arguments for the aggregator

    Returns:
        Aggregator instance

    Example:
        >>> agg = get_aggregator('krum', num_byzantine=2)
        >>> global_model = agg.aggregate(client_updates)
    """
    method = method.lower()

    if method == 'krum':
        return KrumAggregator(**kwargs)
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator(**kwargs)
    elif method == 'median':
        return MedianAggregator()
    elif method in ['fedprox', 'fedavg']:
        return FedProxAggregator(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Choose from: krum, trimmed_mean, median, fedprox, fedavg"
        )
