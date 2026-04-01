"""GPU-accelerated base learners as batched (seeds, N, d) tensors.

Mirrors pynol OGD, OEGD, and OptimisticOGD as GPU-batched variants.

Improvement 2: All N experts for all seeds are one tensor.  The gradient step
and Ball projection are single broadcast operations — no Python for-loop over
N experts.

Three classes are provided (matching the three base learners Sword variants use):
  - BatchedOGD          → mirrors OGD        (used by Ader, SwordSmallLoss)
  - BatchedOEGD         → mirrors OEGD       (used by SwordVariation, SwordBest)
  - BatchedOptimisticOGD→ mirrors OptimisticOGD (used by SwordPP)
"""

import torch
import numpy as np

from pynol_GPU.environment_GPU.domain_GPU import BallGPU


class BatchedOGD:
    """All N OGD experts as one (seeds, N, d) tensor.

    Args:
        step_sizes (array-like): shape (N,) — one step size per expert.
        domain     (BallGPU): Feasible set for batched projection.
        seeds      (int): Number of parallel random seeds.
        X_init     (Tensor, optional): (seeds, N, d) initial state.
        device, dtype: Torch device / dtype.
    """

    def __init__(self, step_sizes, domain: BallGPU, seeds: int,
                 X_init: torch.Tensor = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.step_sizes = torch.tensor(np.asarray(step_sizes),
                                       device=device, dtype=dtype)   # (N,)
        self.domain = domain
        self.seeds  = seeds
        self.N      = len(step_sizes)
        self.d      = domain.dimension
        self.device = device
        self.dtype  = dtype
        self.t      = 0

        self.X = (X_init.to(device=device, dtype=dtype)
                  if X_init is not None
                  else torch.zeros(seeds, self.N, self.d,
                                   device=device, dtype=dtype))

    @property
    def x_bases(self) -> torch.Tensor:
        return self.X                                                  # (seeds, N, d)

    def update(self, grad: torch.Tensor) -> None:
        """Vectorised OGD step.  grad: (seeds, d)."""
        self.X = self.X - self.step_sizes[None, :, None] * grad[:, None, :]
        self.X = self.domain.project_batch(self.X)
        self.t += 1

    def reinit(self, X_init: torch.Tensor = None) -> None:
        if X_init is not None:
            self.X = X_init.to(device=self.device, dtype=self.dtype)
        else:
            self.X = torch.zeros(self.seeds, self.N, self.d,
                                 device=self.device, dtype=self.dtype)
        self.t = 0


class BatchedOEGD:
    """All N OEGD experts as (seeds, N, d) — online extra-gradient descent.

    OEGD maintains two tensors:
      x        = the submitted decision (used for loss/grad)
      middle_x = the "look-ahead" point updated after seeing the gradient

    Update rule (per expert i, seed s):
      x_{t,s,i}       = project(middle_x_{t,s,i} - eta_i * optimism_{t,s,i})
      middle_x_{t+1}   = project(middle_x_{t,s,i} - eta_i * grad_{t,s,i})
      optimism_{t+1}   = grad_at_middle_x   (for next round's opt_by_optimism)

    In the GPU version:
      - optimism_{t,s,i} = grad_{t-1}(middle_x_{t-1,s,i}) stored as self.optimism
      - all ops are (seeds, N, d) broadcasts

    Args:
        step_sizes, domain, seeds, X_init, device, dtype: same as BatchedOGD.
        grad_fn: callable (X: (seeds,N,d), t: int) -> (seeds,N,d) gradients.
                 Provided at construction for the "middle_x gradient" query.
    """

    def __init__(self, step_sizes, domain: BallGPU, seeds: int,
                 X_init: torch.Tensor = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.step_sizes = torch.tensor(np.asarray(step_sizes),
                                       device=device, dtype=dtype)
        self.domain = domain
        self.seeds  = seeds
        self.N      = len(step_sizes)
        self.d      = domain.dimension
        self.device = device
        self.dtype  = dtype
        self.t      = 0

        X0 = (X_init.to(device=device, dtype=dtype)
              if X_init is not None
              else torch.zeros(seeds, self.N, self.d,
                               device=device, dtype=dtype))
        self.X        = X0.clone()
        self.middle_x = X0.clone()
        # optimism for next round: grad at middle_x from last round
        self.optimism = torch.zeros_like(X0)

    @property
    def x_bases(self) -> torch.Tensor:
        return self.X

    def opt_by_optimism(self) -> None:
        """Apply stored optimism to get submitted decision x_t.

        x_t = project(middle_x_t - eta * optimism_t)
        """
        self.X = self.middle_x - self.step_sizes[None, :, None] * self.optimism
        self.X = self.domain.project_batch(self.X)

    def update(self, grad: torch.Tensor, middle_grad: torch.Tensor) -> None:
        """Gradient step on middle_x and store next-round optimism.

        Args:
            grad:        (seeds, d) gradient at meta decision x_t
                         (same for all experts under InnerSurrogate).
            middle_grad: (seeds, N, d) gradient at middle_x points
                         (evaluated by the caller from the loss function).
        """
        # middle_x update: same gradient broadcast over all experts
        self.middle_x = (self.middle_x
                         - self.step_sizes[None, :, None] * grad[:, None, :])
        self.middle_x = self.domain.project_batch(self.middle_x)
        # store gradient at new middle_x as next-round optimism
        self.optimism = middle_grad
        self.t += 1


class BatchedOptimisticOGD:
    """All N OptimisticOGD experts as (seeds, N, d).

    OptimisticOGD maintains:
      x        = project(middle_x - eta * optimism)   [submitted decision]
      middle_x = project(middle_x - eta * grad)        [updated after gradient]

    For SwordPP:
      - optimism comes from LastGradOptimismBase → optimism_{t} = grad_{t-1}
      - grad comes from InnerSurrogateBase → same grad for every expert

    Args: same as BatchedOGD.
    """

    def __init__(self, step_sizes, domain: BallGPU, seeds: int,
                 X_init: torch.Tensor = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.step_sizes = torch.tensor(np.asarray(step_sizes),
                                       device=device, dtype=dtype)
        self.domain = domain
        self.seeds  = seeds
        self.N      = len(step_sizes)
        self.d      = domain.dimension
        self.device = device
        self.dtype  = dtype
        self.t      = 0

        X0 = (X_init.to(device=device, dtype=dtype)
              if X_init is not None
              else torch.zeros(seeds, self.N, self.d,
                               device=device, dtype=dtype))
        self.X        = X0.clone()
        self.middle_x = X0.clone()
        self.optimism = torch.zeros_like(X0)   # (seeds, N, d)

    @property
    def x_bases(self) -> torch.Tensor:
        return self.X

    def opt_by_optimism(self, optimism: torch.Tensor) -> None:
        """Apply external optimism.

        Args:
            optimism: (seeds, d) — LastGrad optimism (same for all experts).
        """
        self.optimism = optimism[:, None, :].expand_as(self.X)
        self.X = self.middle_x - self.step_sizes[None, :, None] * self.optimism
        self.X = self.domain.project_batch(self.X)

    def update(self, grad: torch.Tensor) -> None:
        """Gradient step: middle_x += -eta * grad.

        Args:
            grad: (seeds, d) — InnerSurrogate gradient (same for all experts).
        """
        self.middle_x = (self.middle_x
                         - self.step_sizes[None, :, None] * grad[:, None, :])
        self.middle_x = self.domain.project_batch(self.middle_x)
        self.t += 1

    def reinit(self, X_init: torch.Tensor = None) -> None:
        zeros = torch.zeros(self.seeds, self.N, self.d,
                            device=self.device, dtype=self.dtype)
        X0 = X_init.to(device=self.device, dtype=self.dtype) if X_init is not None else zeros
        self.X        = X0.clone()
        self.middle_x = X0.clone()
        self.optimism = torch.zeros_like(X0)
        self.t = 0
