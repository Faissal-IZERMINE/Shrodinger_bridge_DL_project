"""FID + MSD evaluation pipeline (paper §6.2 Table 2).

Uses ``clean-fid`` to compute the standardised FID. The reference set is the
full 60K MNIST training set; the generated set is produced by running the
forward SDE on EMNIST test inputs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image

from sb.data import get_emnist_test, get_mnist_train_all


# Cache the MNIST reference set across multiple evaluate() calls.
_mnist_train_cache: torch.Tensor | None = None


def tensor_to_png_folder(tensors: torch.Tensor, folder: Path,
                        img_size: int = 32) -> None:
    """Save ``(N, C*H*W)`` flat tensors in [-1, 1] as RGB PNGs (for clean-fid)."""
    folder.mkdir(parents=True, exist_ok=True)
    imgs = tensors.view(-1, 1, img_size, img_size)
    imgs = ((imgs + 1.0) / 2.0 * 255).clamp(0, 255).byte()
    for i, img in enumerate(imgs):
        Image.fromarray(img.squeeze(0).numpy(), mode="L").convert("RGB").save(
            folder / f"{i:05d}.png"
        )


def compute_msd(original: torch.Tensor, generated: torch.Tensor) -> float:
    """Mean Squared Distance between paired flat tensors in [-1, 1]."""
    assert original.shape == generated.shape, "Shape mismatch in MSD inputs."
    return ((original - generated) ** 2).mean(dim=1).mean().item()


def evaluate(bridge_model, device: torch.device, num_steps: int = 30,
            n_samples: int = 4000, batch_size: int = 128) -> dict:
    """Evaluate a bridge checkpoint with FID (clean-fid) and MSD.

    ``bridge_model`` must expose
    ``.sample(x0, method, direction, num_steps) -> (B, T+1, D)`` trajectory.
    Only the final time-step ``[:, -1, :]`` is taken as the generated output.
    """
    global _mnist_train_cache
    from cleanfid import fid

    print("  Loading EMNIST test data ...")
    emnist_test = get_emnist_test(n_samples=n_samples)

    print(f"  Generating {n_samples} MNIST samples ...")
    chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            x0 = emnist_test[start:start + batch_size].to(device)
            traj = bridge_model.sample(
                x0, method="sde", direction="forward", num_steps=num_steps,
            )
            chunks.append(traj[:, -1, :].cpu())
    generated = torch.cat(chunks, dim=0)

    msd = compute_msd(emnist_test, generated)
    print(f"  MSD : {msd:.4f}")

    if _mnist_train_cache is None:
        print("  Loading full MNIST train set (cached for subsequent runs) ...")
        _mnist_train_cache = get_mnist_train_all()

    tmp_root = Path(tempfile.mkdtemp())
    real_dir, fake_dir = tmp_root / "real", tmp_root / "fake"
    try:
        print("  Saving reference and generated images to disk ...")
        tensor_to_png_folder(_mnist_train_cache, real_dir)
        tensor_to_png_folder(generated, fake_dir)
        print("  Computing FID ...")
        fid_score = fid.compute_fid(
            str(real_dir), str(fake_dir), mode="clean", num_workers=4,
        )
        print(f"  FID : {fid_score:.4f}")
    finally:
        shutil.rmtree(tmp_root)

    return {"FID": fid_score, "MSD": msd}
