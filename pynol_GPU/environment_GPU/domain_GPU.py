"""GPU-accelerated domain classes.

Mirrors pynol.environment.domain but operates on PyTorch tensors.

Improvement 1: project_batch handles arbitrary leading batch dims (..., d) via
a single torch.linalg.norm + clamp, replacing the Python loop that called
Ball.project() once per expert.
"""

import torch
import numpy as np


class BallGPU:
    """Euclidean-ball feasible set on a GPU tensor — mirrors pynol Ball.

    Args:
        dimension (int): Feature dimension d.
        radius (float): Ball radius R.
        center (array-like, optional): Ball centre; defaults to origin.
        device (str): Torch device string ('cpu', 'cuda', 'mps').
        dtype (torch.dtype): Floating-point type (default float32).
    """

    def __init__(self, dimension: int, radius: float = 1.0,
                 center=None, device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.dimension = dimension
        self.radius = float(radius)
        self.R = self.radius          # outer-ball radius (mirrors pynol API)
        self.r = self.radius          # inner-ball radius
        self.device = device
        self.dtype = dtype
        self.center = (torch.zeros(dimension, device=device, dtype=dtype)
                       if center is None
                       else torch.tensor(center, device=device, dtype=dtype))

    # ------------------------------------------------------------------
    # Improvement 1: single-vector projection (kept for compatibility)
    # ------------------------------------------------------------------
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project a single vector (d,) onto the ball."""
        diff = x - self.center
        norm = torch.linalg.norm(diff)
        if norm > self.r:
            x = self.center + diff * (self.r / norm)
        return x

    # ------------------------------------------------------------------
    # Improvement 1: vectorised batch projection
    # ------------------------------------------------------------------
    def project_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Project a batch of shape (..., d) onto the ball.

        A single norm call across the last dimension + a broadcast clamp
        replaces the per-expert Python loop.  Works for (N, d),
        (seeds, N, d), or any higher-rank tensor.
        """
        diffs = X - self.center                                      # (..., d)
        norms = torch.linalg.norm(diffs, dim=-1, keepdim=True)       # (..., 1)
        scale = torch.clamp(self.r / norms, max=1.0)                 # (..., 1)
        return self.center + diffs * scale

    def init_x_batch(self, seeds: int, N: int,
                     seed: int = None) -> torch.Tensor:
        """Random initialisation for (seeds, N, d) expert states.

        All N experts for one seed share the same starting point, matching
        the pynol convention where a single seed is passed to DiscreteSSP
        and every base-learner receives the same init.

        Returns:
            Tensor of shape (seeds, N, d).
        """
        if seed is not None:
            torch.manual_seed(seed)
        directions = torch.randn(seeds, self.dimension,
                                 device=self.device, dtype=self.dtype)
        directions = directions / torch.linalg.norm(
            directions, dim=-1, keepdim=True)
        radii = torch.rand(seeds, 1, device=self.device, dtype=self.dtype)
        x0 = self.radius * directions * radii                        # (seeds, d)
        # broadcast over N — all experts per seed start at the same point
        return x0.unsqueeze(1).expand(seeds, N, self.dimension).clone()

    @classmethod
    def from_pynol(cls, ball, device: str = 'cpu',
                   dtype: torch.dtype = torch.float32) -> 'BallGPU':
        """Construct from an existing pynol Ball instance."""
        return cls(ball.dimension, ball.radius, ball.center,
                   device=device, dtype=dtype)
