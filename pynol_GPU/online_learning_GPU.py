"""GPU online-learning runner.

Improvement 8: replaces the outer ``for seed in seeds`` loop in the notebook
with a single (seeds, N, d) tensor batch — all random seeds run in one
forward pass.

Accepts any GPU learner with a .opt(loss_fn, t) -> (seeds,) interface:
  - AderGPU
  - SwordVariationGPU
  - SwordBestGPU
  - SwordPPGPU

Usage (replacing the pynol multiple_online_learning call):

    from pynol_GPU.environment_GPU.loss_function_GPU import SquareLossGPU
    from pynol_GPU.learner_GPU.models_GPU.dynamic.ader_GPU import AderGPU
    from pynol_GPU.learner_GPU.models_GPU.dynamic.swordpp_GPU import SwordPPGPU
    from pynol_GPU.online_learning_GPU import online_learning_GPU

    device  = 'mps'   # or 'cuda' / 'cpu'
    loss_fn = SquareLossGPU(feature, label, scale=scale, device=device)

    # Each learner runs all 5 seeds in one forward pass
    aderpp  = AderGPU(domain=domain, T=T, G=G, surrogate=True,
                      seeds=5, seed=0, device=device)
    swordpp = SwordPPGPU(domain=domain, T=T, G=G, L_smooth=L_smooth,
                         seeds=5, seed=0, device=device)

    cum_ader,   tm = online_learning_GPU(T, loss_fn, aderpp)   # (5, T) numpy
    cum_sword,  tm = online_learning_GPU(T, loss_fn, swordpp)
"""

import time
import numpy as np
import torch


def online_learning_GPU(T: int, loss_fn, learner,
                        return_numpy: bool = True):
    """Run T rounds of online learning for all seeds in parallel.

    Args:
        T            (int): Time horizon.
        loss_fn      (SquareLossGPU): Pre-loaded feature/label tensors.
        learner      : Any GPU learner with .opt(loss_fn, t) -> (seeds,).
        return_numpy (bool): Transfer result to CPU numpy if True (default).

    Returns:
        cumulative_loss: shape (seeds, T) — cumulative loss per seed.
                         numpy array if return_numpy=True, else Tensor.
        tm: (T,) array of per-round wall-clock times (seconds).
    """
    losses = torch.zeros(learner.seeds, T,
                         device=learner.device, dtype=learner.dtype)
    tm = np.zeros(T)

    for t in range(T):
        t0 = time.time()
        losses[:, t] = learner.opt(loss_fn, t)
        tm[t] = time.time() - t0

    cumulative = losses.cumsum(dim=-1)                               # (seeds, T)

    if return_numpy:
        return cumulative.cpu().numpy(), tm
    return cumulative, tm


def multiple_online_learning_GPU(T: int, loss_fn, learners: list,
                                  return_numpy: bool = True):
    """Run multiple GPU learners sequentially and collect results.

    Mirrors pynol multiple_online_learning API.

    Args:
        T        (int): Time horizon.
        loss_fn  (SquareLossGPU): Pre-loaded feature/label tensors.
        learners (list): List of GPU learner instances.

    Returns:
        all_cum_losses: list of (seeds, T) arrays — one per learner.
        all_tm:         list of (T,) timing arrays.
    """
    results = []
    for learner in learners:
        cum, tm = online_learning_GPU(T, loss_fn, learner,
                                      return_numpy=return_numpy)
        results.append((cum, tm))
    return results
