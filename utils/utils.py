import os
import numpy as np
import random
import glob

from collections import OrderedDict, defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Set seeds and flags to ensure full reproducibility across Python,
    NumPy, PyTorch (CPU and CUDA), and hashing for dataloader operations.

    This function configures:
    - Python's built-in random module
    - NumPy random state
    - PyTorch CPU and all available CUDA devices
    - PyTorch deterministic algorithms for CuDNN
    - Python hash seed environment variable

    Args:
        seed (int): The seed value to enforce determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # (Optional) ensure hashing is fixed (e.g., for dataloader shuffling)
    os.environ["PYTHONHASHSEED"] = str(seed)


def average_checkpoints(epoch_list, checkpoint_path_template):
    """
    Average the model parameters from multiple checkpoints.

    This function loads model checkpoints saved at different epochs,
    accumulates their parameter tensors, and computes the mean for each parameter.
    It can be used to smooth out training noise and potentially yield
    a better final model than using a single checkpoint.

    Args:
        epoch_list (list of int): List of epoch numbers whose checkpoints will be averaged.
        checkpoint_path_template (str): Template path to load checkpoints, should
            include a format specifier for the epoch number, e.g. 'checkpoints/epoch_{}.pt'.

    Returns:
        OrderedDict: A state_dict with the same keys as the original model,
        but with parameter values averaged across the given checkpoints.

    Notes:
        - If the checkpoint has a nested dictionary under the key 'model',
          it is assumed that the actual state_dict is under that key.
          Adjust this logic if your checkpoint format differs.
        - Assumes all checkpoints have identical model architectures.
    """
    avg_state_dict = defaultdict(float)
    count = 0

    for epoch in epoch_list:
        path = checkpoint_path_template.format(epoch)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = (
            checkpoint["model"] if "model" in checkpoint else checkpoint
        )  # adjust if needed

        for key, value in state_dict.items():
            avg_state_dict[key] += value

        count += 1

    for key in avg_state_dict:
        avg_state_dict[key] /= count

    return OrderedDict(avg_state_dict)


def cleanup_all_checkpoints(checkpoint_dir="checkpoints", pattern="epoch_*.pt"):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
    for path in checkpoint_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        """
        Stops training if validation loss doesn't improve.
        - patience: Number of epochs to wait before stopping
        - delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True  # NEW: Set stop flag to True


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Sigmoid Focal Loss Implementation.

        Args:
        - alpha (float): Weighting factor for positive class (helps with class imbalance).
        - gamma (float): Focusing parameter (higher = more focus on hard samples).
        - reduction (str): 'mean' for batch-wise average, 'sum' for total loss, 'none' for per sample loss.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes focal loss.

        Args:
        - logits (torch.Tensor): Model raw outputs (before sigmoid).
        - targets (torch.Tensor): Ground truth labels (binary: 0 or 1).

        Returns:
        - torch.Tensor: Computed focal loss.
        """
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        targets = targets.float()

        # Compute the focal loss
        loss_pos = (
            -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs + 1e-8)
        )
        loss_neg = (
            -(1 - self.alpha)
            * (probs**self.gamma)
            * (1 - targets)
            * torch.log(1 - probs + 1e-8)
        )

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def plot_metric_curves(
    train_auc,
    valid_auc,
    output_dir,
    train_precision=None,
    valid_precision=None,
    train_recall=None,
    valid_recall=None,
    title="Training vs Validation Metrics",
):
    epochs = range(1, len(train_auc) + 1)
    plt.figure(figsize=(10, 6))

    # AUC
    plt.plot(epochs, train_auc, label="Train AUC", linewidth=2)
    plt.plot(epochs, valid_auc, label="Valid AUC", linewidth=2, linestyle="--")

    # Optional: precision
    if train_precision and valid_precision:
        plt.plot(epochs, train_precision, label="Train Precision", alpha=0.6)
        plt.plot(
            epochs, valid_precision, label="Valid Precision", alpha=0.6, linestyle="--"
        )

    # Optional: recall
    if train_recall and valid_recall:
        plt.plot(epochs, train_recall, label="Train Recall", alpha=0.6)
        plt.plot(epochs, valid_recall, label="Valid Recall", alpha=0.6, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")
    plt.savefig(
        f"{output_dir}/plots/train_vs_valid_auc_curve.eps", format="eps", dpi=600
    )
