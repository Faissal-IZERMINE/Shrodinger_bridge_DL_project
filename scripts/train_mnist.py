"""Train Curriculum-Enhanced alpha-DSBM on MNIST <-> EMNIST.

Replicates the constrained-compute experiment from paper §6.2.

Example:
    python scripts/train_mnist.py \\
        --pretrain-epochs 213 --finetune-epochs 60 \\
        --base-channels 128 --batch-size 128 \\
        --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make src/ importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from sb.bridge import SchrodingerBridgeMatching, TrainingConfig, set_global_seed
from sb.data import load_mnist_emnist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-epochs", type=int, default=213)
    p.add_argument("--finetune-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=128,
                   help="UNet base channel count. 128 -> ~5.1M parameters.")
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=30, help="SDE rollout steps.")
    p.add_argument("--pretrain-lr", type=float, default=1.5e-4)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--ratio-leap", type=float, default=0.05,
                   help="Leap Scheduler snap ratio (fraction of total finetune "
                        "steps for the cosine ramp).")
    p.add_argument("--ratio-osc", type=float, default=0.75,
                   help="Oscillating Scheduler snap ratio.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-size", type=int, default=62000,
                   help="Number of samples to draw from MNIST/EMNIST.")
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--resume-pretrain", type=str, default=None)
    p.add_argument("--resume-finetune", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None,
                   help="If set, log to this wandb project.")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print("Loading MNIST and EMNIST ...")
    mnist_data, emnist_data = load_mnist_emnist(
        subset_size=args.subset_size, seed=args.seed,
    )
    split_idx = min(60000, len(mnist_data))
    mnist_train, emnist_train = mnist_data[:split_idx], emnist_data[:split_idx]
    print(f"Training samples: {len(mnist_train)}")

    wandb_run = None
    if args.wandb_project is not None:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    plot_fn = None
    if wandb_run is not None:
        from sb.visualization import progress_plot
        plot_fn = progress_plot

    cfg = TrainingConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        steps=args.steps,
        eps=args.eps,
        pretrain_lr=args.pretrain_lr,
        finetune_lr=args.finetune_lr,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        ratio_osc=args.ratio_osc,
        ratio_leap=args.ratio_leap,
        seed=args.seed,
    )

    bridge = SchrodingerBridgeMatching(
        law_0=mnist_train, law_1=emnist_train, cfg=cfg,
        ckpt_dir=args.output_dir, wandb_run=wandb_run, plot_fn=plot_fn,
    )

    fixed_val = torch.tensor(mnist_train[:16], dtype=torch.float32).to(bridge.device)

    print("\n=== Phase 1: Pretraining ===")
    bridge.pretrain(
        fixed_test_batch=fixed_val, print_every=1000,
        save_every_epoch=20, resume_from=args.resume_pretrain,
    )
    bridge.save_checkpoint(os.path.join(args.output_dir, "pretrained_final.pt"))

    print("\n=== Phase 2: Curriculum-Enhanced Finetuning ===")
    bridge.finetune(
        fixed_test_batch=fixed_val, print_every=1000,
        save_every_epoch=5, resume_from=args.resume_finetune,
    )
    bridge.save_checkpoint(os.path.join(args.output_dir, "finetuned_final.pt"))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
