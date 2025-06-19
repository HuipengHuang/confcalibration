import numpy as np
def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''

    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n


def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely
    on the number of classes and the number of examples per class.

    This relies on two heuristics:
    1) We want at least 150 points per cluster on average
    2) We need more samples as we try to distinguish between more distributions.
    To distinguish between 2 distribution, want at least 4 samples per class.
    To distinguish between 5 distributions, want at least 10 samples per class.

    Output: n_clustering, num_clusters

    '''
    # Alias for convenience
    N = n_totalcal
    K = num_classes

    n_clustering = int(N * K / (75 + K))
    num_clusters = int(np.floor(n_clustering / 2))
    return n_clustering, num_clusters


import torch
import numpy as np
from typing import Tuple


def compute_calibration_metrics(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
        norm: str = 'l1'
) -> Tuple[float, float, float, float, float]:
    """
    Compute ECE, ACE, MCE, and PIECE for a classifier.

    Args:
        logits: Model logits (shape: [N, K]).
        labels: Ground truth labels (shape: [N]).
        n_bins: Number of bins for binning-based metrics.
        norm: 'l1' for ECE/ACE/MCE, 'l2' for squared error (PIECE).

    Returns:
        Tuple of (ECE, ACE, MCE, PIECE).
    """

    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)

    # ------ Binning-based metrics (ECE, ACE, MCE) ------
    # Uniform binning (ECE)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Adaptive binning (ACE) - equal sample count per bin
    sorted_indices = torch.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    bin_size = len(labels) // n_bins

    ece, ace, mce = 0.0, 0.0, 0.0

    for i in range(n_bins):
        # --- ECE ---
        in_bin = (confidences >= bin_lowers[i]) & (confidences < bin_uppers[i])
        bin_acc = accuracies[in_bin].float().mean() if in_bin.any() else 0.0
        bin_conf = confidences[in_bin].mean() if in_bin.any() else 0.0
        bin_weight = in_bin.float().sum() / len(labels)
        ece += bin_weight * torch.abs(bin_acc - bin_conf)

        # --- ACE ---
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(labels)
        bin_acc_adaptive = sorted_accuracies[start:end].float().mean()
        bin_conf_adaptive = sorted_confidences[start:end].mean()
        ace += torch.abs(bin_acc_adaptive - bin_conf_adaptive) / n_bins

        # --- MCE ---
        current_error = torch.abs(bin_acc - bin_conf)
        if current_error > mce:
            mce = current_error

    # ------ PIECE (Plugin Estimator) ------
    piece = torch.mean((confidences - accuracies.float()) ** 2)

    return torch.mean(accuracies).item(),ece.item(), ace.item(), mce.item(), piece.item()
