"""GPU-accelerated loss functions.

Mirrors pynol.environment.loss_function but operates on PyTorch tensors.

Improvement 3: loss(X, t) computes X @ phi[t] as a single matmul, replacing
N individual dot products when X has a batch leading dimension.

Improvement 4: grad(X, t) returns the gradient for any shape (..., d) in one
broadcast: (X@phi - y) * phi.  For Ader++ the gradient is the SAME for every
expert (LinearSurrogate), so the caller only needs grad(x_meta) — a (seeds, d)
tensor — then broadcasts over N without any extra computation.
"""

import torch
import numpy as np


class SquareLossGPU:
    """Batched square loss mirroring pynol SquareLoss.

    f_t(x) = scale/2 * (phi_t · x  -  y_t)^2

    Args:
        feature (array-like): shape (T, d) — feature matrix for all rounds.
        label   (array-like): shape (T,)  — labels for all rounds.
        scale   (float): Loss scale coefficient.
        device  (str): Torch device string.
        dtype   (torch.dtype): Floating-point type.
    """

    def __init__(self, feature, label, scale: float = 1.0,
                 device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.phi   = torch.tensor(np.asarray(feature),
                                  device=device, dtype=dtype)   # (T, d)
        self.y     = torch.tensor(np.asarray(label),
                                  device=device, dtype=dtype)   # (T,)
        self.scale = scale
        self.device = device
        self.dtype  = dtype

    # ------------------------------------------------------------------
    # Improvement 3: batched loss — one matmul instead of N dot products
    # ------------------------------------------------------------------
    def loss(self, X: torch.Tensor, t: int) -> torch.Tensor:
        """Compute loss for X of shape (..., d).  Returns (...) tensor.

        `X @ phi[t]` is a single BLAS call regardless of leading dims.
        """
        phi = self.phi[t]                    # (d,)
        y   = self.y[t]                      # scalar
        scores = X @ phi                     # (...,)
        return self.scale * 0.5 * (scores - y) ** 2

    # ------------------------------------------------------------------
    # Improvement 4: batched gradient — one matmul + outer broadcast
    # ------------------------------------------------------------------
    def grad(self, X: torch.Tensor, t: int) -> torch.Tensor:
        """Compute gradient for X of shape (..., d).  Returns (..., d) tensor.

        For Ader++ (LinearSurrogate) every expert uses the same gradient as
        the meta decision, so X = x_meta has shape (seeds, d) and grad()
        is called exactly once per round.
        """
        phi = self.phi[t]                            # (d,)
        y   = self.y[t]                              # scalar
        residuals = self.scale * (X @ phi - y)       # (...,)
        return residuals.unsqueeze(-1) * phi          # (..., d)

    @classmethod
    def from_pynol(cls, loss_fn, device: str = 'cpu',
                   dtype: torch.dtype = torch.float32) -> 'SquareLossGPU':
        """Construct from an existing pynol SquareLoss instance."""
        return cls(loss_fn.feature, loss_fn.label, loss_fn.scale,
                   device=device, dtype=dtype)
