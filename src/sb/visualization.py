"""Sampling-progress visualisation utility used during training.

Decoupled from ``bridge.py`` so the core training module has no matplotlib
import. Pass ``progress_plot`` as the ``plot_fn`` argument to
``SchrodingerBridgeMatching``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torchvision


def _unnorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)


def progress_plot(bridge, fixed_source_samples: torch.Tensor,
                  title: str = "Progress", img_size: int = 32):
    """2x2 grid: SDE input/output, ODE input/output. Returns the Figure."""
    B = fixed_source_samples.shape[0]
    with torch.no_grad():
        traj_sde = bridge.sample(fixed_source_samples, method="sde", num_steps=30)
        traj_ode = bridge.sample(fixed_source_samples, method="ode", num_steps=30)
    final_sde, final_ode = traj_sde[:, -1, :], traj_ode[:, -1, :]

    def grid(x):
        return torchvision.utils.make_grid(
            _unnorm(x.reshape(B, 1, img_size, img_size))[:16], nrow=4, padding=2,
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(grid(fixed_source_samples).permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Input (Fixed Source) - SDE")
    axes[0, 1].imshow(grid(final_sde).permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title(f"{title} Output - SDE")
    axes[1, 0].imshow(grid(fixed_source_samples).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("Input (Fixed Source) - ODE")
    axes[1, 1].imshow(grid(final_ode).permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title(f"{title} Output - ODE")
    plt.close(fig)
    return fig
