"""
Attack Models for Testing Federated Learning Robustness

Implements various Byzantine attacks to test the robustness of federated
learning systems:
- Label Flipping: Flip labels to poison training data
- Model Scaling: Scale model updates to disrupt aggregation
- Backdoor Attack: Inject backdoor triggers into models
- Gaussian Noise: Add random noise to model updates

These attacks simulate malicious clients in federated learning scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from collections import OrderedDict
import copy


class LabelFlippingAttack:
    """
    Label Flipping Attack: Flip labels to poison the training data.

    For binary classification:
    - 0 -> 1, 1 -> 0

    For multi-class:
    - Can flip to random class or specific target class

    This simulates malicious clients providing incorrect labels.
    """

    def __init__(self, flip_ratio: float = 0.5, target_class: Optional[int] = None):
        """
        Args:
            flip_ratio: Fraction of labels to flip (0.0 to 1.0)
            target_class: If set, flip all labels to this class (targeted attack)
                         If None, flip randomly or invert (untargeted attack)
        """
        assert 0 <= flip_ratio <= 1, "flip_ratio must be in [0, 1]"
        self.flip_ratio = flip_ratio
        self.target_class = target_class

    def apply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply label flipping attack to data.

        Args:
            X: Features (not modified)
            y: Labels to flip
            num_classes: Number of classes (for multi-class, auto-detected if None)

        Returns:
            (X, poisoned_y): Original features and flipped labels
        """
        y_poisoned = y.copy()

        # Determine which samples to poison
        num_samples = len(y)
        num_poison = int(num_samples * self.flip_ratio)
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)

        # Detect if binary or multi-class
        unique_classes = np.unique(y)
        if num_classes is None:
            num_classes = len(unique_classes)

        # Apply attack
        if self.target_class is not None:
            # Targeted attack: flip to specific class
            y_poisoned[poison_indices] = self.target_class
        else:
            # Untargeted attack
            if num_classes == 2:
                # Binary: flip 0->1, 1->0
                y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]
            else:
                # Multi-class: flip to random other class
                for idx in poison_indices:
                    current_class = y[idx]
                    other_classes = [c for c in unique_classes if c != current_class]
                    y_poisoned[idx] = np.random.choice(other_classes)

        return X, y_poisoned


class ModelScalingAttack:
    """
    Model Scaling Attack: Scale model updates to disrupt aggregation.

    Multiplies the model update by a large factor to dominate the aggregation.
    This can cause divergence or bias the global model.

    Reference: Fang et al. "Local model poisoning attacks to byzantine-robust
               federated learning" (USENIX Security 2020)
    """

    def __init__(self, scaling_factor: float = 10.0):
        """
        Args:
            scaling_factor: Multiplier for model updates (typically > 1)
                           Large values (e.g., 10-100) cause more disruption
        """
        self.scaling_factor = scaling_factor

    def apply(self, model_update: OrderedDict) -> OrderedDict:
        """
        Scale all parameters in the model update.

        Args:
            model_update: Model state dict from malicious client

        Returns:
            Scaled model update
        """
        scaled_update = OrderedDict()

        for param_name, param in model_update.items():
            if isinstance(param, torch.Tensor):
                scaled_update[param_name] = param * self.scaling_factor
            else:
                scaled_update[param_name] = param * self.scaling_factor

        return scaled_update


class BackdoorAttack:
    """
    Backdoor Attack: Inject a backdoor trigger into the model.

    The attacker modifies data by adding a specific pattern (trigger), and
    associates it with a target label. The global model will then misclassify
    any input with the trigger as the target class.

    Example trigger: Set specific features to 1 or add a pattern

    Reference: Bagdasaryan et al. "How to backdoor federated learning"
               (AISTATS 2020)
    """

    def __init__(
        self,
        trigger_indices: List[int],
        trigger_value: float = 1.0,
        target_class: int = 1,
        poison_ratio: float = 0.1
    ):
        """
        Args:
            trigger_indices: Feature indices to modify as trigger
            trigger_value: Value to set for trigger features
            target_class: Class label for backdoored samples
            poison_ratio: Fraction of training data to backdoor
        """
        self.trigger_indices = trigger_indices
        self.trigger_value = trigger_value
        self.target_class = target_class
        self.poison_ratio = poison_ratio

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject backdoor into training data.

        Args:
            X: Features
            y: Labels

        Returns:
            (backdoored_X, backdoored_y): Poisoned data and labels
        """
        num_samples = len(X)
        num_poison = int(num_samples * self.poison_ratio)

        # Create backdoored data
        X_backdoor = X.copy()
        y_backdoor = y.copy()

        # Select samples to backdoor (random or all)
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)

        # Inject trigger
        for idx in poison_indices:
            X_backdoor[idx, self.trigger_indices] = self.trigger_value
            y_backdoor[idx] = self.target_class

        return X_backdoor, y_backdoor

    def create_backdoor_test_set(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test set with backdoor trigger to evaluate attack success rate.

        Args:
            X_test: Clean test features
            y_test: Clean test labels (optional)

        Returns:
            (backdoored_X_test, target_labels): All test samples with trigger
        """
        X_backdoor_test = X_test.copy()

        # Inject trigger into all test samples
        X_backdoor_test[:, self.trigger_indices] = self.trigger_value

        # All should be classified as target class
        if y_test is not None:
            y_backdoor_test = np.full_like(y_test, self.target_class)
        else:
            y_backdoor_test = np.full(len(X_test), self.target_class)

        return X_backdoor_test, y_backdoor_test


class GaussianNoiseAttack:
    """
    Gaussian Noise Attack: Add random Gaussian noise to model updates.

    This is a simple untargeted attack that adds noise to disrupt training.
    Less sophisticated than scaling attacks but can still degrade performance.
    """

    def __init__(self, noise_std: float = 0.1):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
        """
        self.noise_std = noise_std

    def apply(self, model_update: OrderedDict) -> OrderedDict:
        """
        Add Gaussian noise to model parameters.

        Args:
            model_update: Model state dict

        Returns:
            Noisy model update
        """
        noisy_update = OrderedDict()

        for param_name, param in model_update.items():
            if isinstance(param, torch.Tensor):
                noise = torch.randn_like(param) * self.noise_std
                noisy_update[param_name] = param + noise
            else:
                noise = np.random.randn(*param.shape) * self.noise_std
                noisy_update[param_name] = param + noise

        return noisy_update


class SignFlippingAttack:
    """
    Sign Flipping Attack: Flip the sign of model updates.

    Instead of moving towards minimizing loss, the malicious client moves
    in the opposite direction to maximize loss.

    This is particularly effective against simple averaging.
    """

    def apply(self, model_update: OrderedDict) -> OrderedDict:
        """
        Flip the sign of all model parameters.

        Args:
            model_update: Model state dict

        Returns:
            Sign-flipped model update
        """
        flipped_update = OrderedDict()

        for param_name, param in model_update.items():
            if isinstance(param, torch.Tensor):
                flipped_update[param_name] = -param
            else:
                flipped_update[param_name] = -param

        return flipped_update


# Utility class to coordinate attacks
class AttackCoordinator:
    """
    Coordinates multiple attacks on malicious clients.

    Helps simulate realistic scenarios where some clients are malicious
    and apply different attack strategies.
    """

    def __init__(
        self,
        num_clients: int,
        malicious_ratio: float,
        attack_type: str = 'label_flipping',
        **attack_kwargs
    ):
        """
        Args:
            num_clients: Total number of clients
            malicious_ratio: Fraction of malicious clients (0.0 to 1.0)
            attack_type: Type of attack ('label_flipping', 'model_scaling',
                        'backdoor', 'gaussian_noise', 'sign_flipping')
            **attack_kwargs: Arguments for the attack class
        """
        self.num_clients = num_clients
        self.num_malicious = int(num_clients * malicious_ratio)

        # Select malicious clients randomly
        self.malicious_clients = set(
            np.random.choice(num_clients, self.num_malicious, replace=False)
        )

        # Create attack instance
        if attack_type == 'label_flipping':
            self.attack = LabelFlippingAttack(**attack_kwargs)
        elif attack_type == 'model_scaling':
            self.attack = ModelScalingAttack(**attack_kwargs)
        elif attack_type == 'backdoor':
            self.attack = BackdoorAttack(**attack_kwargs)
        elif attack_type == 'gaussian_noise':
            self.attack = GaussianNoiseAttack(**attack_kwargs)
        elif attack_type == 'sign_flipping':
            self.attack = SignFlippingAttack()
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        self.attack_type = attack_type

    def is_malicious(self, client_id: int) -> bool:
        """Check if a client is malicious."""
        return client_id in self.malicious_clients

    def apply_data_attack(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data poisoning attack if client is malicious.

        Args:
            client_id: Client ID
            X: Features
            y: Labels
            **kwargs: Additional arguments for attack

        Returns:
            (X, y): Possibly poisoned data
        """
        if client_id in self.malicious_clients:
            if self.attack_type in ['label_flipping', 'backdoor']:
                return self.attack.apply(X, y, **kwargs)

        return X, y

    def apply_model_attack(
        self,
        client_id: int,
        model_update: OrderedDict
    ) -> OrderedDict:
        """
        Apply model poisoning attack if client is malicious.

        Args:
            client_id: Client ID
            model_update: Model state dict

        Returns:
            Possibly poisoned model update
        """
        if client_id in self.malicious_clients:
            if self.attack_type in ['model_scaling', 'gaussian_noise', 'sign_flipping']:
                return self.attack.apply(model_update)

        return model_update


# Quick test function
if __name__ == "__main__":
    print("Testing Attack Models...")

    # Test Label Flipping
    print("\n[1] Label Flipping Attack:")
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    attack = LabelFlippingAttack(flip_ratio=0.3)
    X_poison, y_poison = attack.apply(X, y)
    print(f"   Original labels: {y[:10]}")
    print(f"   Poisoned labels: {y_poison[:10]}")
    print(f"   Flip rate: {(y != y_poison).mean():.2%}")

    # Test Model Scaling
    print("\n[2] Model Scaling Attack:")
    model = OrderedDict({
        'weight': torch.randn(5, 10),
        'bias': torch.randn(5)
    })
    attack = ModelScalingAttack(scaling_factor=10.0)
    scaled_model = attack.apply(model)
    print(f"   Original norm: {torch.norm(model['weight']):.4f}")
    print(f"   Scaled norm: {torch.norm(scaled_model['weight']):.4f}")

    # Test Attack Coordinator
    print("\n[3] Attack Coordinator:")
    coordinator = AttackCoordinator(
        num_clients=10,
        malicious_ratio=0.3,
        attack_type='label_flipping',
        flip_ratio=0.5
    )
    print(f"   Malicious clients: {sorted(coordinator.malicious_clients)}")
    print(f"   Client 0 malicious: {coordinator.is_malicious(0)}")

    print("\n[OK] All attack tests passed!")
