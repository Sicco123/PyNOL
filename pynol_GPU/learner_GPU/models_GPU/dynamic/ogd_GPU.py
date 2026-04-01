"""OGDGPU — GPU-accelerated Online Gradient Descent baseline.

Mirrors pynol OGD (base.py).  In the GPU world this is BatchedOGD with N=1
(single expert, fixed step size), so the (seeds, 1, d) tensor collapses to
effectively (seeds, d) — but we keep the same shape convention for consistency
with the runner.

Usage (replacing [OGD(..., seed=s) for s in seeds]):

    from pynol_GPU.learner_GPU.models_GPU.dynamic.ogd_GPU import OGDGPU

    ogd = OGDGPU(domain=domain, step_size=min_step_size,
                 seeds=5, seed=0, device=device)
    cum_loss, tm = online_learning_GPU(T, loss_fn, ogd)   # (5, T)
"""

import numpy as np
import torch

from pynol_GPU.environment_GPU.domain_GPU import BallGPU
from pynol_GPU.environment_GPU.loss_function_GPU import SquareLossGPU
from pynol_GPU.learner_GPU.base_GPU import BatchedOGD


class OGDGPU:
    """OGD for all seeds in parallel on GPU.

    Args:
        domain    (BallGPU | Ball): Feasible set.
        step_size (float): Fixed step size eta.
        seeds     (int): Number of parallel random initialisations.
        seed      (int, optional): Master random seed for initialisation.
        device, dtype: Torch device / dtype.
    """

    def __init__(self, domain, step_size: float,
                 seeds: int = 1, seed: int = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):

        if not isinstance(domain, BallGPU):
            domain = BallGPU.from_pynol(domain, device=device, dtype=dtype)

        # N=1: one expert per seed, shape (seeds, 1, d)
        X_init = domain.init_x_batch(seeds=seeds, N=1, seed=seed)

        self.schedule = BatchedOGD(
            step_sizes=np.array([step_size]),
            domain=domain,
            seeds=seeds,
            X_init=X_init,
            device=device,
            dtype=dtype)

        self.d      = domain.dimension
        self.seeds  = seeds
        self.device = device
        self.dtype  = dtype
        self.t      = 0

    def opt(self, loss_fn: SquareLossGPU, t: int) -> torch.Tensor:
        """One round for all seeds.

        Returns:
            loss: (seeds,) origin losses at the current decision.
        """
        # x shape: (seeds, 1, d) → squeeze to (seeds, d) for loss/grad
        x = self.schedule.x_bases[:, 0, :]                           # (seeds, d)

        loss = loss_fn.loss(x, t)                                     # (seeds,)
        grad = loss_fn.grad(x, t)                                     # (seeds, d)

        # BatchedOGD.update expects (seeds, d) and broadcasts over N=1
        self.schedule.update(grad)

        self.t += 1
        return loss
