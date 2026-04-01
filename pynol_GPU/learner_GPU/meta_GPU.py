"""GPU-accelerated meta-learners: HedgeGPU and OptimisticHedgeGPU.

Mirrors pynol.learner.meta.Hedge and OptimisticHedge.

Improvement 5: probability vectors live as (seeds, N) GPU tensors.
All multiplicative weight updates are single element-wise expressions
evaluated in log-space for numerical stability — no Python loops, no
scalar numpy ops.

OptimisticHedge adds a two-step update:
  p_t  = middle_prob * exp(-lr * M_t) / Z          [opt_by_optimism]
  middle_prob_t+1 = middle_prob * exp(-lr * L_t) / Z  [update]

with an adaptive OptimisticLR that adjusts the learning rate from
the optimism/loss gap (mirrors pynol OptimisticLR).
"""

import torch
import numpy as np


class HedgeGPU:
    """Hedge algorithm for all seeds in parallel.

    Args:
        N      (int): Number of experts.
        lr     (array-like): shape (T,) — learning-rate schedule.
        seeds  (int): Number of parallel random seeds.
        prior  (str | array-like): 'nonuniform' (pynol default) or 'uniform'.
        device (str): Torch device string.
        dtype  (torch.dtype): Floating-point type.
    """

    def __init__(self, N: int, lr, seeds: int = 1,
                 prior='nonuniform', device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.N      = N
        self.seeds  = seeds
        self.device = device
        self.dtype  = dtype
        self.t      = 0

        self.lr = torch.tensor(np.asarray(lr),
                               device=device, dtype=dtype)           # (T,)

        prob0 = self._init_prob(N, prior)
        self.prob = (prob0.unsqueeze(0)
                        .expand(seeds, N)
                        .clone()
                        .to(device=device, dtype=dtype))             # (seeds, N)

    def update(self, loss_bases: torch.Tensor) -> None:
        """Hedge multiplicative weight update in log-space.

        Args:
            loss_bases: (seeds, N) per-expert loss this round.
        """
        lr_t = self.lr[self.t]
        log_prob = torch.log(self.prob) - lr_t * loss_bases
        log_prob = log_prob - log_prob.logsumexp(dim=-1, keepdim=True)
        self.prob = torch.exp(log_prob)
        self.t += 1

    @staticmethod
    def _init_prob(N: int, prior) -> torch.Tensor:
        if isinstance(prior, str) and prior == 'nonuniform':
            w = torch.tensor(
                [(N + 1) / (N * i * (i + 1)) for i in range(1, N + 1)],
                dtype=torch.float64)
            return (w / w.sum()).float()
        elif isinstance(prior, str) and prior == 'uniform':
            return torch.ones(N) / N
        else:
            w = torch.tensor(np.asarray(prior), dtype=torch.float32)
            return w / w.sum()


class OptimisticHedgeGPU:
    """Optimistic Hedge with adaptive learning rate for all seeds in parallel.

    Mirrors pynol OptimisticHedge + OptimisticLR.

    The two-step update is:
        p_t       = middle_prob * exp(-lr_t * M_t) / Z      [opt_by_optimism]
        middle_prob_{t+1} = middle_prob * exp(-lr_t * (L_t + correction)) / Z  [update]

    The adaptive LR follows pynol's OptimisticLR:
        lr_{t+1} = min(lr_upper, 1 / (8 * sum_{s<=t} (L_s - M_s)^2))

    Args:
        N           (int): Number of experts.
        lr_upper    (float): Upper bound on learning rate.
        seeds       (int): Number of parallel random seeds.
        device      (str): Torch device string.
        dtype       (torch.dtype): Floating-point type.
    """

    def __init__(self, N: int, lr_upper: float, seeds: int = 1,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.N         = N
        self.seeds     = seeds
        self.lr_upper  = lr_upper
        self.device    = device
        self.dtype     = dtype
        self.t         = 0

        prob0 = torch.ones(N, dtype=dtype) / N
        self.prob = (prob0.unsqueeze(0)
                        .expand(seeds, N)
                        .clone()
                        .to(device=device, dtype=dtype))             # (seeds, N)
        self.middle_prob = self.prob.clone()

        # Adaptive LR: one scalar shared across seeds (mirrors pynol)
        self.lr      = torch.tensor(lr_upper, device=device, dtype=dtype)
        # Cumulative squared (loss - optimism) per seed × expert for LR update
        self._cum_sq = torch.zeros(seeds, N, device=device, dtype=dtype)
        self._optimism = torch.zeros(seeds, N, device=device, dtype=dtype)

    def opt_by_optimism(self, optimism_meta: torch.Tensor) -> None:
        """Apply optimism to get the current-round probability.

        p_t = middle_prob * exp(-lr * M_t) / Z

        Args:
            optimism_meta: (seeds, N) optimism signal.
        """
        self._optimism = optimism_meta
        log_prob = (torch.log(self.middle_prob)
                    - self.lr * optimism_meta)
        log_prob = log_prob - log_prob.logsumexp(dim=-1, keepdim=True)
        self.prob = torch.exp(log_prob)

    def update(self, loss_bases: torch.Tensor) -> None:
        """Optimistic Hedge update step (log-space).

        middle_prob_{t+1} = middle_prob * exp(-lr * L_t) / Z

        Then update the adaptive LR.

        Args:
            loss_bases: (seeds, N) per-expert loss this round.
        """
        log_mp = (torch.log(self.middle_prob)
                  - self.lr * loss_bases)
        log_mp = log_mp - log_mp.logsumexp(dim=-1, keepdim=True)
        self.middle_prob = torch.exp(log_mp)

        # Update adaptive LR: lr = min(lr_upper, 1/(8*sum(L-M)^2))
        self._cum_sq = self._cum_sq + (loss_bases - self._optimism) ** 2
        # Use the mean over seeds×experts as a single shared lr (matches pynol scalar)
        denom = 8.0 * self._cum_sq.mean()
        if denom > 0:
            self.lr = torch.clamp(
                torch.tensor(1.0 / denom.item(),
                             device=self.device, dtype=self.dtype),
                max=self.lr_upper)
        self.t += 1
