"""Evaluate one or more checkpoints with FID and MSD.

Example:
    python scripts/eval_checkpoints.py \\
        --checkpoints \\
            checkpoints/pretrained_final.pt=Pretrained \\
            checkpoints/finetune/step_56000.pt=Finetune-epoch-30 \\
            checkpoints/finetune/step_112091.pt=Finetune-epoch-60 \\
        --n-samples 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sb.bridge import SchrodingerBridgeMatching, TrainingConfig
from sb.eval import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        help='Checkpoints to evaluate in the form path=label '
             '(e.g. checkpoints/pretrained_final.pt=Pretrained)',
    )
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a bridge with the right architecture; we'll overwrite its weights
    # from each checkpoint. Dataset content doesn't matter for evaluation;
    # use a tiny placeholder.
    cfg = TrainingConfig(base_channels=args.base_channels)
    placeholder = np.zeros((2, 1024), dtype=np.float32)
    bridge = SchrodingerBridgeMatching(
        law_0=placeholder, law_1=placeholder, cfg=cfg,
    )

    all_results = {}
    for entry in args.checkpoints:
        if "=" in entry:
            path, name = entry.split("=", 1)
        else:
            path, name = entry, Path(entry).stem

        print(f"\n{'='*60}\n  Evaluating: {name}\n  Checkpoint: {path}\n{'='*60}")
        bridge.load_checkpoint(path)
        all_results[name] = evaluate(
            bridge_model=bridge, device=device,
            num_steps=args.num_steps, n_samples=args.n_samples,
            batch_size=args.batch_size,
        )

    print(f"\n{'='*60}\n{'Model':<35} {'FID':>10} {'MSD':>10}\n{'-'*60}")
    for name, res in all_results.items():
        print(f"{name:<35} {res['FID']:>10.4f} {res['MSD']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
