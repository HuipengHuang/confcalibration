import datetime
import torch
import numpy as np
import random
import os
from torch import Tensor


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_exp_result(args, result_dict, path=None):
    current_time = datetime.datetime.now()
    month_day = current_time.strftime("%b%d")
    if path is None:
        path = f"./experiment/{args.algorithm}"
    name = f"{args.dataset}_{args.model}_{args.loss}loss"

    save_path = os.path.join(path, month_day)
    save_path = os.path.join(save_path, args.score)

    i = 0
    while True:
        if os.path.exists(os.path.join(save_path, f"{name}_{i}")):
            i += 1
        else:
            break
    folder_path = os.path.join(save_path, f"{name}_{i}")
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "result.txt"), "w") as f:
        for key in result_dict.keys():
            f.write(f"{key}: {result_dict[key]}\n")

        f.write("\nDetailed Setup \n")

        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                f.write(f"{k}: {v}\n")


def neural_sort(
        scores: Tensor,
        tau: float = 0.1,
) -> Tensor:
    """
    Soft sorts scores (descending) along last dimension
    Follows implementation form
    https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py

    Grover, Wang et al., Stochastic Optimization of Sorting Networks via Continuous Relaxations

    Args:
        scores (Tensor): scores to sort
        tau (float, optional): smoothness factor. Defaults to 0.01.
        hard (bool, optional): whether to hard sort. Defaults to False.

    Returns:
        Tensor: permutation matrix such that sorted_scores = P @ scores
    """
    A = (scores[..., :, None] - scores[..., None, :]).abs()
    n = scores.shape[-1]

    B = A @ torch.ones(n, 1, device=A.device)
    C = scores[..., :, None] * (n - 1 - 2 * torch.arange(n, device=A.device, dtype=torch.float))
    P_scores = (C - B).transpose(-2, -1)
    P_hat = torch.softmax(P_scores / tau, dim=-1)

    return P_hat
def soft_quantile(
        scores: Tensor,
        q: float,
        dim=-1,
        **kwargs
) -> Tensor:
    # swap requested dim with final dim
    dims = list(range(len(scores.shape)))
    dims[-1], dims[dim] = dims[dim], dims[-1]
    scores = scores.permute(*dims)
    # normalize scores on last dimension
    scores_norm = (scores - scores.mean()) / 3. * scores.std()
    # obtain permutation matrix for scores
    P_hat = neural_sort(scores_norm, **kwargs)
    # use permutation matrix to sort scores
    sorted_scores = (P_hat @ scores[..., None])[..., 0]
    # turn quantiles into indices to select
    n = scores.shape[-1]
    squeeze = False
    if isinstance(q, float):
        squeeze = True
        q = [q]
    q = torch.tensor(q, dtype=torch.float, device=scores.device)
    indices = (1 - q) * (n + 1) - 1
    indices_low = torch.floor(indices).long()
    indices_frac = indices - indices_low
    indices_high = indices_low + 1
    # select quantiles from computed scores:
    quantiles = sorted_scores[..., torch.cat([indices_low, indices_high])]
    quantiles = quantiles[..., :q.shape[0]] + indices_frac * (quantiles[..., q.shape[0]:] - quantiles[..., :q.shape[0]])
    # restore dimension order
    if len(dims) > 1:
        quantiles = quantiles.permute(*dims)

    if squeeze:
        quantiles = quantiles.squeeze(dim)

    return quantiles