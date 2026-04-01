"""SwordPPGPU — GPU-accelerated Sword++ with multi-seed parallelism.

Mirrors pynol SwordPP (swordpp.py).

Algorithm design:
  Base learner:   OptimisticOGD
  Meta:           OptimisticHedge (adaptive LR)
  Surrogate base: InnerSurrogateBase   → l_i = <X_i, grad>
  Optimism base:  LastGradOptimismBase → m_t = grad_{t-1}  (same for all experts)
  Surrogate meta: InnerSwitchingSurrogateMeta
                  → l_meta_i = <X_i, grad> + penalty * ||X_i - X_i_prev||^2
  Optimism meta:  InnerSwitchingOptimismMeta
                  → M_meta_i = <X_i_new, optimism> + penalty * ||X_i_new - X_i||^2

All formulas are inlined as einsum / broadcast ops on (seeds, N, d) tensors.
No separate specification folder is needed.
"""

import numpy as np
import torch

from pynol_GPU.environment_GPU.domain_GPU import BallGPU
from pynol_GPU.environment_GPU.loss_function_GPU import SquareLossGPU
from pynol_GPU.learner_GPU.base_GPU import BatchedOptimisticOGD
from pynol_GPU.learner_GPU.meta_GPU import OptimisticHedgeGPU


def _build_step_pool(min_step: float, max_step: float,
                     grid: float = 2.0) -> np.ndarray:
    pool = [min_step]
    s = min_step
    while s <= max_step:
        s *= grid
        pool.append(s)
    return np.array(pool)


class SwordPPGPU:
    """Sword++ running all seeds in parallel on GPU.

    Drop-in replacement for a list of pynol SwordPP instances.

    Args:
        domain        (BallGPU | Ball): Feasible set.
        T             (int): Time horizon.
        G             (float): Gradient bound.
        L_smooth      (float): Smoothness constant.
        min_step_size, max_step_size (float, optional): Step-size pool bounds.
        seeds         (int): Number of parallel random initialisations.
        seed          (int, optional): Master random seed.
        device, dtype : Torch device / dtype.
    """

    def __init__(self, domain, T: int, G: float, L_smooth: float,
                 min_step_size: float = None, max_step_size: float = None,
                 seeds: int = 1, seed: int = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):

        if not isinstance(domain, BallGPU):
            domain = BallGPU.from_pynol(domain, device=device, dtype=dtype)

        D = 2.0 * domain.R
        if min_step_size is None:
            min_step_size = (D**2 / (G**2 * T)) ** 0.5
        if max_step_size is None:
            max_step_size = 1.0 / (8.0 * L_smooth)

        step_pool = _build_step_pool(min_step_size, max_step_size)
        N         = len(step_pool)
        lr_upper  = 1.0 / (8.0 * D**2 * L_smooth)
        self.penalty = 2.0 * L_smooth

        X_init = domain.init_x_batch(seeds=seeds, N=N, seed=seed)

        self.schedule = BatchedOptimisticOGD(
            step_sizes=step_pool, domain=domain, seeds=seeds,
            X_init=X_init, device=device, dtype=dtype)

        self.meta = OptimisticHedgeGPU(
            N=N, lr_upper=lr_upper, seeds=seeds,
            device=device, dtype=dtype)

        self.N      = N
        self.d      = domain.dimension
        self.seeds  = seeds
        self.device = device
        self.dtype  = dtype
        self.t      = 0

        # LastGradOptimismBase: optimism for next round = current grad
        self._last_grad = torch.zeros(seeds, domain.dimension,
                                      device=device, dtype=dtype)
        # InnerSwitchingSurrogateMeta needs X from previous round
        self._X_prev = X_init.clone()

    def opt(self, loss_fn: SquareLossGPU, t: int) -> torch.Tensor:
        """One round for all seeds.

        Returns:
            loss: (seeds,) origin losses.
        """
        X_before_opt = self.schedule.x_bases                         # (seeds, N, d)

        # ---- opt_by_optimism for base: x_t = project(mid - eta * last_grad) ----
        # LastGradOptimismBase: optimism = last grad (same for all experts)
        self.schedule.opt_by_optimism(self._last_grad)

        X    = self.schedule.x_bases                                  # (seeds, N, d) after opt
        prob = self.meta.prob                                         # (seeds, N)

        # ---- InnerSwitchingOptimismMeta: M_i = <X_new_i, optimism> + penalty*||X_new_i - X_old_i||^2 ----
        # optimism here is schedule.optimism = _last_grad broadcast over experts
        opt_broadcast = self._last_grad[:, None, :]                   # (seeds, 1, d)
        inner_part    = torch.einsum('snd,snd->sn', X, opt_broadcast.expand_as(X))  # (seeds, N)
        diff          = X - self._X_prev                              # (seeds, N, d)
        switch_part   = self.penalty * diff.pow(2).sum(dim=-1)        # (seeds, N)
        optimism_meta = inner_part + switch_part                      # (seeds, N)
        self.meta.opt_by_optimism(optimism_meta)

        # ---- meta decision ----
        prob = self.meta.prob
        x    = torch.einsum('sn,snd->sd', prob, X)                   # (seeds, d)
        loss = loss_fn.loss(x, t)                                     # (seeds,)
        grad = loss_fn.grad(x, t)                                     # (seeds, d)

        # ---- InnerSurrogateBase: base grad = grad (same for all experts) ----
        # ---- InnerSwitchingSurrogateMeta: l_i = <X_i, grad> + penalty*||X_i - X_prev_i||^2 ----
        inner_loss  = torch.einsum('snd,sd->sn', X, grad)             # (seeds, N)
        switch_loss = self.penalty * (X - self._X_prev).pow(2).sum(dim=-1)  # (seeds, N)
        loss_bases  = inner_loss + switch_loss                         # (seeds, N)

        # ---- Store current X as previous for next round ----
        self._X_prev = X.detach().clone()

        # ---- Gradient update for base learners (InnerSurrogate → same grad) ----
        self.schedule.update(grad)

        # ---- Meta update ----
        self.meta.update(loss_bases)

        # ---- LastGradOptimismBase: store grad for next round ----
        self._last_grad = grad.detach().clone()

        self.t += 1
        return loss
