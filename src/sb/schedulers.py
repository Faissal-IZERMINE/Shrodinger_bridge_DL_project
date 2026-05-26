"""Curriculum schedulers for Curriculum-Enhanced alpha-DSBM (paper §5.1).

Both schedulers are stateful: call ``step()`` once per training step. After
the snap point they return their terminal value forever (1.0 for Leap; 0.5
for Oscillating).
"""

from __future__ import annotations

import math

import torch


class LeapScheduler:
    """Leap Scheduler lambda(n) (paper Eq. for lambda).

    Cosine ramp from ``start_lambda`` to 1 over the first ``ratio * total_steps``
    steps, then constant 1.

        lambda(n) = start_lambda + (1 - start_lambda) * (1 - cos(pi * tau)) / 2

    where tau = min(1, n / n_snap) and n_snap = floor(ratio * total_steps).
    Used to blend true X_0/X_1 with EMA-generated X_tilde_0/X_tilde_1 during
    finetuning: X_blended = lerp(X_true, X_tilde, lambda). Early training uses
    self-generated targets (easier curriculum); later training "leaps" toward
    the true distribution.
    """

    def __init__(self, total_steps: int, ratio: float = 0.4,
                 start_lambda: float = 0.0) -> None:
        self.total_steps = total_steps
        self.start_lambda = start_lambda
        self.snap_step = int(total_steps * ratio)
        self.step_idx = 0

    def step(self) -> float:
        n = self.step_idx
        self.step_idx += 1
        if n >= self.snap_step:
            return 1.0
        progress = min(1.0, n / self.snap_step) if self.snap_step > 0 else 1.0
        cos_val = 0.5 * (1 - math.cos(math.pi * progress))
        return self.start_lambda + (1.0 - self.start_lambda) * cos_val


class OscillatingScheduler:
    """Oscillating Scheduler (alpha_osc, beta_osc) with exponentially decaying
    amplitude around 0.5 (paper §5.1).

        alpha_osc(n) = clamp(0.5 + 0.5 * exp(-decay * tau) * sin(2 pi freq * tau + phi), 0, 1)
        beta_osc(n)  = 1 - alpha_osc(n)

    with tau = n / n_snap, n_snap = floor(ratio * total_steps), and phi a
    uniform random phase in [0, 2 pi) drawn at init. After n_snap, both weights
    are fixed at 0.5. Used as forward/backward loss weights in the curriculum
    finetuning loss L = alpha_osc * fwd + beta_osc * bwd.
    """

    def __init__(self, total_steps: int, ratio: float = 0.6,
                 freq: float = 20.0, decay: float = 5.0,
                 device: str | torch.device = "cpu",
                 phi: torch.Tensor | None = None) -> None:
        self.total_steps = total_steps
        self.snap_step = int(total_steps * ratio)
        self.freq = freq
        self.decay = decay
        self.device = device
        self.phi = phi if phi is not None else torch.rand(1, device=device) * 2 * math.pi
        self.step_idx = 0

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.step_idx
        self.step_idx += 1
        if n >= self.snap_step or self.snap_step == 0:
            alpha = torch.tensor(0.5, device=self.device)
        else:
            tau = n / self.snap_step
            amplitude = 0.5 * torch.exp(torch.tensor(-self.decay * tau, device=self.device))
            alpha = 0.5 + amplitude * torch.sin(
                2 * math.pi * self.freq * tau + self.phi
            )
            alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha, 1.0 - alpha

    def state_dict(self) -> dict:
        return {"step_idx": self.step_idx, "phi": self.phi.detach().cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.step_idx = state["step_idx"]
        self.phi = state["phi"].to(self.device)
