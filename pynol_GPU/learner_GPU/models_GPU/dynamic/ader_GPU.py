"""AderGPU — GPU-accelerated Ader / Ader++ with multi-seed parallelism.

Mirrors pynol.learner.models.dynamic.ader.Ader but operates entirely on
PyTorch tensors, combining all previous improvements in a single opt() call.

Improvement 6: surrogate losses for all seeds × experts computed with a
single einsum: X_dot_g = einsum('snd,sd->sn', X, grad).

Improvement 7: meta aggregation x = prob @ X_bases is a single einsum:
x = einsum('sn,snd->sd', prob, X).

Improvement 8: the (seeds, N, d) batch dimension eliminates the outer
``for seed in seeds`` loop in the notebook entirely — all seeds run in
one forward pass.
"""

import numpy as np
import torch

from pynol_GPU.environment_GPU.domain_GPU import BallGPU
from pynol_GPU.environment_GPU.loss_function_GPU import SquareLossGPU
from pynol_GPU.learner_GPU.base_GPU import BatchedOGD
from pynol_GPU.learner_GPU.meta_GPU import HedgeGPU


def _build_step_pool(min_step: float, max_step: float,
                     grid: float = 2.0) -> np.ndarray:
    """Discrete step-size pool — mirrors DiscreteSSP.discretize."""
    pool = [min_step]
    s = min_step
    while s <= max_step:
        s *= grid
        pool.append(s)
    return np.array(pool)


class AderGPU:
    """Ader / Ader++ running all seeds in parallel on a single GPU batch.

    Drop-in replacement for a *list* of pynol Ader instances (one per seed).
    Instead of ``[Ader(..., seed=s) for s in seeds]`` use one AderGPU with
    ``seeds=len(seeds)``.

    Args:
        domain        (BallGPU | Ball): Feasible set.  A pynol Ball is
                      accepted and converted automatically.
        T             (int): Time horizon.
        G             (float): Gradient bound.
        surrogate     (bool): True → Ader++, False → standard Ader.
        min_step_size (float, optional): Smallest expert step size.
        max_step_size (float, optional): Largest expert step size.
        seeds         (int): Number of parallel random initialisations.
        seed          (int, optional): Master random seed for init.
        device        (str): Torch device ('cpu', 'cuda', 'mps').
        dtype         (torch.dtype): Floating-point type.
    """

    def __init__(self, domain, T: int, G: float,
                 surrogate: bool = True,
                 min_step_size: float = None,
                 max_step_size: float = None,
                 seeds: int = 1,
                 seed: int = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):

        # Accept pynol Ball or BallGPU
        if not isinstance(domain, BallGPU):
            domain = BallGPU.from_pynol(domain, device=device, dtype=dtype)

        D = 2.0 * domain.R
        if min_step_size is None:
            min_step_size = D / G * (7.0 / (2.0 * T)) ** 0.5
        if max_step_size is None:
            max_step_size = D / G * (7.0 / (2.0 * T) + 2.0) ** 0.5

        step_pool = _build_step_pool(min_step_size, max_step_size, grid=2)
        N  = len(step_pool)
        lr = np.array([1.0 / (G * D * (t + 1) ** 0.5) for t in range(T)])

        # Improvement 8: initialise (seeds, N, d) state for all seeds at once
        X_init = domain.init_x_batch(seeds=seeds, N=N, seed=seed)   # (seeds, N, d)

        self.schedule = BatchedOGD(                # Improvement 2
            step_sizes=step_pool,
            domain=domain,
            seeds=seeds,
            X_init=X_init,
            device=device,
            dtype=dtype)

        self.meta = HedgeGPU(                      # Improvement 5
            N=N, lr=lr, seeds=seeds,
            prior='nonuniform',
            device=device, dtype=dtype)

        self.surrogate = surrogate
        self.T         = T
        self.N         = N
        self.d         = domain.dimension
        self.seeds     = seeds
        self.device    = device
        self.dtype     = dtype
        self.t         = 0

    def opt(self, loss_fn: SquareLossGPU, t: int) -> torch.Tensor:
        """Run one round for all seeds simultaneously.

        Args:
            loss_fn: SquareLossGPU holding the full feature/label tensors.
            t:       Current round index.

        Returns:
            loss: (seeds,) origin loss at the meta decision for each seed.
        """
        X    = self.schedule.x_bases                               # (seeds, N, d)
        prob = self.meta.prob                                      # (seeds, N)

        # Improvement 7: meta aggregation — single einsum over seeds × experts
        x = torch.einsum('sn,snd->sd', prob, X)                   # (seeds, d)

        # Improvement 3+4: loss and gradient at meta decision
        loss = loss_fn.loss(x, t)                                  # (seeds,)
        grad = loss_fn.grad(x, t)                                  # (seeds, d)

        # Compute per-expert losses for the Hedge update
        if self.surrogate:
            # Ader++: LinearSurrogate  →  l_i = <X_i - x, grad>
            # Improvement 6: one einsum replaces N inner products
            X_dot_g = torch.einsum('snd,sd->sn', X, grad)         # (seeds, N)
            x_dot_g = torch.einsum('sd,sd->s',   x, grad)         # (seeds,)
            loss_bases = X_dot_g - x_dot_g.unsqueeze(1)           # (seeds, N)
        else:
            # Standard Ader: origin loss at each expert's point
            # Improvement 3: one einsum → all expert scores at once
            phi      = loss_fn.phi[t]                              # (d,)
            y        = loss_fn.y[t]                                # scalar
            X_scores = torch.einsum('snd,d->sn', X, phi)          # (seeds, N)
            loss_bases = loss_fn.scale * 0.5 * (X_scores - y)**2  # (seeds, N)

        # Improvement 2: vectorised OGD update + batch projection
        self.schedule.update(grad)

        # Improvement 5: Hedge multiplicative weight update
        self.meta.update(loss_bases)

        self.t += 1
        return loss                                                 # (seeds,)
