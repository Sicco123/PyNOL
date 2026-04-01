"""SwordGPU — GPU-accelerated Sword variants with multi-seed parallelism.

Mirrors pynol SwordVariation, SwordBest (sword.py) as GPU batched versions.

Algorithm      Base learner    Meta          Surrogate meta      Optimism meta
-----------    ------------    ----          --------------      -------------
SwordVariation OEGD            OptimHedge    Inner               SwordVariation
SwordBest      OEGD+OGD        OptimHedge    Inner               SwordBest

Both use InnerSurrogateMeta → l_i = <X_i, grad>  (einsum 'snd,sd->sn').

SwordVariation optimism meta:
    M_t,i = <X_i, grad(x_t-1)>  where x_t-1 = sum_i p_{t-1,i} X_i
    → stored as self._last_grad_meta and broadcast-dotted with X each round.

SwordBest optimism meta:
    Learns best of {M^v = <X, grad(x_t-1)>, M^s = 0} via an inner Hedge(2).
    → self._inner_hedge: (seeds, 2) probability over two optimism types.
"""

import numpy as np
import torch

from pynol_GPU.environment_GPU.domain_GPU import BallGPU
from pynol_GPU.environment_GPU.loss_function_GPU import SquareLossGPU
from pynol_GPU.learner_GPU.base_GPU import BatchedOEGD, BatchedOGD
from pynol_GPU.learner_GPU.meta_GPU import OptimisticHedgeGPU


def _build_step_pool(min_step: float, max_step: float,
                     grid: float = 2.0) -> np.ndarray:
    pool = [min_step]
    s = min_step
    while s <= max_step:
        s *= grid
        pool.append(s)
    return np.array(pool)


# ---------------------------------------------------------------------------
# SwordVariationGPU
# ---------------------------------------------------------------------------

class SwordVariationGPU:
    """Sword with gradient-variation bound, all seeds in parallel.

    Mirrors pynol SwordVariation.

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
            min_step_size = (D**2 / (8.0 * G**2 * T)) ** 0.5
        if max_step_size is None:
            max_step_size = 1.0 / (4.0 * L_smooth)

        step_pool = _build_step_pool(min_step_size, max_step_size)
        N  = len(step_pool)
        lr_upper = 1.0 / (8.0 * D**2 * L_smooth)

        X_init = domain.init_x_batch(seeds=seeds, N=N, seed=seed)

        self.schedule = BatchedOEGD(
            step_sizes=step_pool, domain=domain, seeds=seeds,
            X_init=X_init, device=device, dtype=dtype)

        self.meta = OptimisticHedgeGPU(
            N=N, lr_upper=lr_upper, seeds=seeds,
            device=device, dtype=dtype)

        self.d         = domain.dimension
        self.seeds     = seeds
        self.device    = device
        self.dtype     = dtype
        self.t         = 0

        # SwordVariation optimism: grad from previous round's meta decision
        self._last_grad_meta = torch.zeros(seeds, domain.dimension,
                                           device=device, dtype=dtype)

    def opt(self, loss_fn: SquareLossGPU, t: int) -> torch.Tensor:
        """One round for all seeds.

        Returns:
            loss: (seeds,) origin losses.
        """
        X    = self.schedule.x_bases                                  # (seeds, N, d)
        prob = self.meta.prob                                         # (seeds, N)

        # --- SwordVariation optimism meta: M_{t,i} = <X_i, last_grad_meta> ---
        optimism_meta = torch.einsum('snd,sd->sn',
                                     X, self._last_grad_meta)         # (seeds, N)
        self.meta.opt_by_optimism(optimism_meta)

        # --- opt_by_optimism for OEGD experts ---
        self.schedule.opt_by_optimism()

        # --- meta decision and loss ---
        prob = self.meta.prob
        x    = torch.einsum('sn,snd->sd', prob, X)                   # (seeds, d)
        loss = loss_fn.loss(x, t)                                     # (seeds,)
        grad = loss_fn.grad(x, t)                                     # (seeds, d)

        # Store for next round's optimism
        self._last_grad_meta = grad.detach().clone()

        # --- InnerSurrogateMeta: l_i = <X_i, grad> ---
        loss_bases = torch.einsum('snd,sd->sn', X, grad)              # (seeds, N)

        # --- middle_x gradient for OEGD (grad at each middle_x point) ---
        # For InnerSurrogate, base gradient = meta gradient (same grad)
        # OEGD needs grad at middle_x to set next-round optimism.
        # Under InnerSurrogate, grad is constant over x, so middle_grad = grad.
        middle_grad = grad[:, None, :].expand_as(X)                   # (seeds, N, d)

        # --- Updates ---
        self.schedule.update(grad, middle_grad)
        self.meta.update(loss_bases)

        self.t += 1
        return loss


# ---------------------------------------------------------------------------
# SwordBestGPU
# ---------------------------------------------------------------------------

class SwordBestGPU:
    """Sword with best-of-both-worlds bound, all seeds in parallel.

    Mirrors pynol SwordBest.

    Combines two step-size pools:
      - OEGD pool  (gradient-variation experts)
      - OGD  pool  (small-loss experts)

    Optimism meta: learns beta_t ∈ [0,1] weighting M^v vs M^s = 0 via
    a per-seed Hedge(2) over the two candidate optimisms.

    Args: same as SwordVariationGPU.
    """

    def __init__(self, domain, T: int, G: float, L_smooth: float,
                 min_step_size: float = None, max_step_size: float = None,
                 seeds: int = 1, seed: int = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):

        if not isinstance(domain, BallGPU):
            domain = BallGPU.from_pynol(domain, device=device, dtype=dtype)

        D = 2.0 * domain.R
        # Two separate step-size pools (mirrors pynol SwordBest)
        min_oegd = (D**2 / (8.0 * G**2 * T)) ** 0.5 if min_step_size is None else min_step_size
        max_oegd = 1.0 / (4.0 * L_smooth)              if max_step_size is None else max_step_size
        min_ogd  = (D / (16.0 * L_smooth * G * T)) ** 0.5 if min_step_size is None else min_step_size
        max_ogd  = 1.0 / (4.0 * L_smooth)               if max_step_size is None else max_step_size

        pool_oegd = _build_step_pool(min_oegd, max_oegd)
        pool_ogd  = _build_step_pool(min_ogd,  max_ogd)
        N_oegd = len(pool_oegd)
        N_ogd  = len(pool_ogd)
        N      = N_oegd + N_ogd
        lr_upper = 1.0 / (8.0 * D**2 * L_smooth)

        X_init_oegd = domain.init_x_batch(seeds=seeds, N=N_oegd, seed=seed)
        X_init_ogd  = domain.init_x_batch(seeds=seeds, N=N_ogd,  seed=seed)

        self.sched_oegd = BatchedOEGD(
            step_sizes=pool_oegd, domain=domain, seeds=seeds,
            X_init=X_init_oegd, device=device, dtype=dtype)

        self.sched_ogd = BatchedOGD(
            step_sizes=pool_ogd, domain=domain, seeds=seeds,
            X_init=X_init_ogd, device=device, dtype=dtype)

        self.meta = OptimisticHedgeGPU(
            N=N, lr_upper=lr_upper, seeds=seeds,
            device=device, dtype=dtype)

        self.N_oegd    = N_oegd
        self.N         = N
        self.d         = domain.dimension
        self.seeds     = seeds
        self.device    = device
        self.dtype     = dtype
        self.t         = 0

        # Inner Hedge(2) per seed for SwordBest optimism — (seeds, 2)
        self._inner_prob = (torch.ones(seeds, 2, device=device, dtype=dtype)
                            / 2.0)
        self._inner_cum_loss = torch.zeros(seeds, 2, device=device, dtype=dtype)
        self._last_grad_meta  = torch.zeros(seeds, self.d,
                                            device=device, dtype=dtype)
        self._last_grad_meta_prev = torch.zeros(seeds, self.d,
                                                device=device, dtype=dtype)

    def opt(self, loss_fn: SquareLossGPU, t: int) -> torch.Tensor:
        """One round for all seeds.

        Returns:
            loss: (seeds,) origin losses.
        """
        X_oegd = self.sched_oegd.x_bases                             # (seeds, N_oegd, d)
        X_ogd  = self.sched_ogd.x_bases                              # (seeds, N_ogd,  d)
        X      = torch.cat([X_oegd, X_ogd], dim=1)                   # (seeds, N, d)
        prob   = self.meta.prob                                       # (seeds, N)

        # --- SwordBest optimism meta ---
        # M^v = last grad at meta decision; M^s = 0
        # beta_t = _inner_prob[:, 0];  optimism = beta * M^v + (1-beta) * 0
        beta  = self._inner_prob[:, 0]                                # (seeds,)
        opt_v = torch.einsum('snd,sd->sn', X, self._last_grad_meta)  # (seeds, N)
        optimism_meta = beta[:, None] * opt_v                         # (seeds, N)
        self.meta.opt_by_optimism(optimism_meta)

        # opt_by_optimism for OEGD experts
        self.sched_oegd.opt_by_optimism()

        # --- meta decision ---
        prob = self.meta.prob
        x    = torch.einsum('sn,snd->sd', prob, X)                   # (seeds, d)
        loss = loss_fn.loss(x, t)                                     # (seeds,)
        grad = loss_fn.grad(x, t)                                     # (seeds, d)

        # --- InnerSurrogateMeta: l_i = <X_i, grad> ---
        loss_bases = torch.einsum('snd,sd->sn', X, grad)              # (seeds, N)

        # --- Update inner Hedge for SwordBest optimism ---
        # loss[0] = ||grad - M^v||^2, loss[1] = ||grad||^2  (per seed)
        l_v = (grad - self._last_grad_meta_prev).pow(2).sum(dim=-1)   # (seeds,)
        l_s = grad.pow(2).sum(dim=-1)                                  # (seeds,)
        inner_loss = torch.stack([l_v, l_s], dim=-1)                  # (seeds, 2)
        self._inner_cum_loss = self._inner_cum_loss + inner_loss
        # Lazy Hedge update (matches pynol's lazy Hedge for the inner meta)
        exp_loss = torch.exp(-2.0 * self._inner_cum_loss)
        self._inner_prob = exp_loss / exp_loss.sum(dim=-1, keepdim=True)

        # Store grads for next round
        self._last_grad_meta_prev = self._last_grad_meta.clone()
        self._last_grad_meta      = grad.detach().clone()

        # --- OEGD middle_x gradient (constant under InnerSurrogate) ---
        middle_grad_oegd = grad[:, None, :].expand_as(X_oegd)
        self.sched_oegd.update(grad, middle_grad_oegd)
        self.sched_ogd.update(grad)
        self.meta.update(loss_bases)

        self.t += 1
        return loss
