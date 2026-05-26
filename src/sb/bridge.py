"""Curriculum-Enhanced alpha-DSBM trainer (paper Algorithm 1).

Wraps the drift network, EMA copy, dataloaders, pretraining loop, and the
curriculum finetuning loop into a single class. Pretraining is identical to
the original alpha-DSBM (Appendix A of the paper); the contribution lives
in the finetuning loop, which adds:

- Leap Scheduler:  X_blended = lerp(X_true, X_tilde, lambda)
- Oscillating Scheduler:  L = alpha_osc * fwd_loss + beta_osc * bwd_loss
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sb.data import LawDataset
from sb.models import EnhancedUNet
from sb.schedulers import LeapScheduler, OscillatingScheduler


@dataclass
class TrainingConfig:
    """All hyperparameters for a Curriculum-Enhanced alpha-DSBM run."""
    pretrain_epochs: int = 213
    finetune_epochs: int = 60
    batch_size: int = 128
    steps: int = 30
    eps: float = 1.0
    pretrain_lr: float = 1.5e-4
    finetune_lr: float = 1e-4
    ema_decay: float = 0.999
    base_channels: int = 128
    num_workers: int = 2
    ratio_osc: float = 0.75
    ratio_leap: float = 0.05
    eps_time: float = 0.001
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    seed: int = 42


@dataclass
class LossHistory:
    pretrain_total: list[float] = field(default_factory=list)
    pretrain_forward: list[float] = field(default_factory=list)
    pretrain_backward: list[float] = field(default_factory=list)
    finetune_total: list[float] = field(default_factory=list)
    finetune_forward: list[float] = field(default_factory=list)
    finetune_backward: list[float] = field(default_factory=list)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SchrodingerBridgeMatching:
    """Curriculum-Enhanced alpha-DSBM trainer.

    Args:
        law_0, law_1: source / target samples as ``(N, D)`` arrays or tensors.
        cfg: ``TrainingConfig`` hyperparameters.
        ckpt_dir: where to save pretrain / finetune checkpoints.
        wandb_run: optional active wandb run for logging.
        plot_fn: optional callable ``(bridge, fixed_test_batch, title) -> Figure``
            invoked every ``print_every`` steps. Decoupled so the bridge core
            has no matplotlib import.
    """

    def __init__(self, law_0, law_1, cfg: TrainingConfig,
                 ckpt_dir: str = "checkpoints", wandb_run=None,
                 plot_fn=None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.plot_fn = plot_fn

        law_0_t = torch.as_tensor(law_0, dtype=torch.float32)
        law_1_t = torch.as_tensor(law_1, dtype=torch.float32)
        self.Law_0 = LawDataset(law_0_t)
        self.Law_1 = LawDataset(law_1_t)
        self.dim = law_0_t.shape[-1]

        self.T = 1.0
        self.delta_t = self.T / cfg.steps
        self.n_steps = cfg.steps
        self.t_list = [
            torch.tensor([t], device=self.device)
            for t in torch.linspace(0.0, self.T, self.n_steps + 1)
        ]

        self.Bs = cfg.batch_size
        self.bs = cfg.batch_size // 2
        self.global_step = 0
        self.criterion = nn.MSELoss()
        self.loss_history = LossHistory()

        self.v_theta = EnhancedUNet(base_channels=cfg.base_channels).to(self.device)
        self.ema_model = copy.deepcopy(self.v_theta).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.pretrain_dir = os.path.join(ckpt_dir, "pretrain")
        self.finetune_dir = os.path.join(ckpt_dir, "finetune")
        os.makedirs(self.pretrain_dir, exist_ok=True)
        os.makedirs(self.finetune_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # EMA & dataloaders
    # ------------------------------------------------------------------
    def _update_ema(self) -> None:
        decay = self.cfg.ema_decay
        with torch.no_grad():
            for p, ema_p in zip(self.v_theta.parameters(),
                                self.ema_model.parameters()):
                ema_p.mul_(decay).add_(p.data, alpha=1 - decay)

    def get_ema_model(self) -> nn.Module:
        return self.ema_model

    def _make_loaders(self, epoch: int, batch_size: int, pretrain: bool = True):
        # Independent shuffles for law_0 and law_1 (unpaired training).
        offset_0 = 42 if pretrain else 75
        offset_1 = 99 if pretrain else 156

        def make(dataset, offset):
            g = torch.Generator().manual_seed(offset + epoch)

            def worker_init_fn(worker_id):
                np.random.seed(offset + epoch * 1000 + worker_id)
                random.seed(offset + epoch * 1000 + worker_id)

            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                generator=g, worker_init_fn=worker_init_fn,
                num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True,
            )

        return make(self.Law_0, offset_0), make(self.Law_1, offset_1)

    # ------------------------------------------------------------------
    # Bridge interpolation utilities
    # ------------------------------------------------------------------
    def _sample_t(self, batch_size: int) -> torch.Tensor:
        eps = self.cfg.eps_time
        return eps + (1 - 2 * eps) * torch.rand(batch_size, 1, device=self.device)

    def _bridge_interp(self, X0, X1, t):
        Z = torch.randn(X0.shape[0], self.dim, device=self.device)
        return (1 - t) * X0 + t * X1 + torch.sqrt(self.cfg.eps * t * (1 - t)) * Z

    # ------------------------------------------------------------------
    # Pretraining (identical to original alpha-DSBM)
    # ------------------------------------------------------------------
    def pretrain(self, fixed_test_batch=None, print_every: int = 500,
                 save_every_epoch: int = 1, resume_from: str | None = None) -> None:
        self.v_theta.train()
        optimizer = torch.optim.Adam(self.v_theta.parameters(), lr=self.cfg.pretrain_lr)
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from, optimizer=optimizer)

        fwd = torch.ones(self.bs, device=self.device)
        bwd = torch.zeros(self.Bs - self.bs, device=self.device)

        for epoch in range(start_epoch, self.cfg.pretrain_epochs):
            loader_0, loader_1 = self._make_loaders(epoch, self.Bs)
            pbar = tqdm(
                zip(loader_0, loader_1),
                desc=f"Pretrain {epoch+1}/{self.cfg.pretrain_epochs}",
            )
            sums = {"loss": 0.0, "fwd": 0.0, "bwd": 0.0, "n": 0}

            for X0, X1 in pbar:
                X0, X1 = X0.to(self.device), X1.to(self.device)
                bs = min(X0.shape[0], X1.shape[0])
                X0, X1 = X0[:bs], X1[:bs]

                t = self._sample_t(bs)
                Xt = self._bridge_interp(X0, X1, t)
                half = bs // 2

                optimizer.zero_grad()
                v_f = self.v_theta(Xt[:half], t[:half], fwd[:half])
                tgt_f = (X1[:half] - Xt[:half]) / torch.clamp(1 - t[:half], min=1e-3)
                loss_f = self.criterion(v_f, tgt_f)

                v_b = self.v_theta(Xt[half:], 1.0 - t[half:], bwd[:bs - half])
                tgt_b = (X0[half:] - Xt[half:]) / torch.clamp(t[half:], min=1e-3)
                loss_b = self.criterion(v_b, tgt_b)

                loss = 0.5 * (loss_f + loss_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_theta.parameters(),
                                               max_norm=self.cfg.grad_clip_norm)
                optimizer.step()
                self._update_ema()

                sums["loss"] += loss.item(); sums["fwd"] += loss_f.item()
                sums["bwd"] += loss_b.item(); sums["n"] += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                self._log_pretrain(loss, loss_f, loss_b, v_f, v_b, epoch,
                                   print_every, fixed_test_batch)
                self.global_step += 1

            if epoch % save_every_epoch == 0:
                self.save_checkpoint(
                    f"{self.pretrain_dir}/step_{self.global_step}.pt",
                    optimizer=optimizer, epoch=epoch,
                )
            self.loss_history.pretrain_total.append(sums["loss"] / sums["n"])
            self.loss_history.pretrain_forward.append(sums["fwd"] / sums["n"])
            self.loss_history.pretrain_backward.append(sums["bwd"] / sums["n"])

    # ------------------------------------------------------------------
    # Curriculum-Enhanced finetuning (paper Algorithm 1)
    # ------------------------------------------------------------------
    def finetune(self, fixed_test_batch=None, print_every: int = 500,
                 save_every_epoch: int = 1, resume_from: str | None = None) -> None:
        self.v_theta.train()
        optimizer = torch.optim.Adam(self.v_theta.parameters(), lr=self.cfg.finetune_lr)

        steps_per_epoch = max(
            len(DataLoader(self.Law_0, batch_size=self.bs)),
            len(DataLoader(self.Law_1, batch_size=self.bs)),
        )
        total_steps = self.cfg.finetune_epochs * steps_per_epoch
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)

        warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                          total_iters=warmup_steps)
        rest = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, rest],
                                    milestones=[warmup_steps])

        leap = LeapScheduler(total_steps=total_steps, ratio=self.cfg.ratio_leap)
        osc = OscillatingScheduler(total_steps=total_steps,
                                   ratio=self.cfg.ratio_osc, device=self.device)

        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(
                resume_from, optimizer=optimizer, lr_scheduler=lr_scheduler,
                leap_scheduler=leap, osc_scheduler=osc,
            )

        for epoch in range(start_epoch, self.cfg.finetune_epochs):
            loader_0, loader_1 = self._make_loaders(epoch, self.bs, pretrain=False)
            pbar = tqdm(
                zip(loader_0, loader_1),
                desc=f"Finetune {epoch+1}/{self.cfg.finetune_epochs}",
            )
            sums = {"loss": 0.0, "fwd": 0.0, "bwd": 0.0, "n": 0}

            for X0, X1 in pbar:
                X0, X1 = X0.to(self.device), X1.to(self.device)
                bs = min(X0.shape[0], X1.shape[0])
                X0, X1 = X0[:bs], X1[:bs]
                fwd_ones = torch.ones(bs, device=self.device)
                bwd_zeros = torch.zeros(bs, device=self.device)

                # 1) Generate EMA targets X_tilde_0, X_tilde_1.
                X1_tilde = self._rollout(X0, fwd_ones, direction="forward", bs=bs)
                X0_tilde = self._rollout(X1, bwd_zeros, direction="backward", bs=bs)
                self.v_theta.train()

                # 2) Curriculum variables.
                lam = leap.step()
                alpha, beta = osc.step()

                X0_blended = torch.lerp(X0, X0_tilde, lam)
                X1_blended = torch.lerp(X1, X1_tilde, lam)

                # 3) Forward / backward losses against blended targets.
                t_f, t_b = self._sample_t(bs), self._sample_t(bs)
                Xt_f = self._bridge_interp(X0_blended, X1, t_f)
                Xt_b = self._bridge_interp(X0, X1_blended, t_b)
                tgt_f = (X1 - Xt_f) / torch.clamp(1 - t_f, min=1e-3)
                tgt_b = (X0 - Xt_b) / torch.clamp(t_b, min=1e-3)

                optimizer.zero_grad()
                v_f = self.v_theta(Xt_f, t_f, fwd_ones)
                loss_f = self.criterion(v_f, tgt_f)
                v_b = self.v_theta(Xt_b, 1.0 - t_b, bwd_zeros)
                loss_b = self.criterion(v_b, tgt_b)

                loss = alpha * loss_f + beta * loss_b
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_theta.parameters(),
                                               max_norm=self.cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                self._update_ema()

                sums["loss"] += loss.item(); sums["fwd"] += loss_f.item()
                sums["bwd"] += loss_b.item(); sums["n"] += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                self._log_finetune(loss, loss_f, loss_b, v_f, v_b, lr_scheduler,
                                   alpha, beta, lam, epoch, print_every,
                                   fixed_test_batch)
                self.global_step += 1

            if epoch % save_every_epoch == 0:
                self.save_checkpoint(
                    f"{self.finetune_dir}/step_{self.global_step}.pt",
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    leap_scheduler=leap, osc_scheduler=osc, epoch=epoch,
                )
            self.loss_history.finetune_total.append(sums["loss"] / sums["n"])
            self.loss_history.finetune_forward.append(sums["fwd"] / sums["n"])
            self.loss_history.finetune_backward.append(sums["bwd"] / sums["n"])

    # ------------------------------------------------------------------
    # SDE rollouts used for sampling and for finetune EMA-target generation
    # ------------------------------------------------------------------
    def _rollout(self, x0: torch.Tensor, s_vec: torch.Tensor, direction: str,
                 bs: int | None = None) -> torch.Tensor:
        """Single SDE rollout from x0 using the EMA model. Last step is
        deterministic to avoid noise at the boundary."""
        model = self.get_ema_model()
        model.eval()
        bs = bs if bs is not None else x0.shape[0]
        x = x0.clone()
        with torch.no_grad():
            t_iter = self.t_list[:-1] if direction == "forward" else reversed(self.t_list[1:])
            for i, t in enumerate(t_iter):
                t_tensor = t.expand(bs, 1)
                t_in = t_tensor if direction == "forward" else 1.0 - t_tensor
                drift = model(x, t_in, s_vec)
                x = x + self.delta_t * drift
                if i < self.n_steps - 1:
                    x = x + np.sqrt(self.cfg.eps * self.delta_t) * torch.randn_like(x)
        return x

    def sample_sde(self, x0: torch.Tensor, direction: str = "forward",
                   num_steps: int | None = None) -> torch.Tensor:
        """Return the full SDE trajectory ``(B, T+1, D)``."""
        model = self.get_ema_model()
        model.eval()
        n = num_steps if num_steps is not None else self.cfg.steps
        delta_t = self.T / n
        t_list = [torch.tensor([t], device=self.device)
                  for t in torch.linspace(0.0, self.T, n + 1)]
        s_vec = (torch.ones(x0.shape[0], device=self.device)
                 if direction == "forward"
                 else torch.zeros(x0.shape[0], device=self.device))

        x = x0.clone()
        trajectory = [x.clone()]
        with torch.no_grad():
            t_iter = t_list[:-1] if direction == "forward" else reversed(t_list[1:])
            for i, t in enumerate(t_iter):
                t_in = t.expand(x.shape[0], 1)
                if direction == "backward":
                    t_in = 1.0 - t_in
                drift = model(x, t_in, s_vec)
                x = x + delta_t * drift
                if i < n - 1:
                    x = x + np.sqrt(self.cfg.eps * delta_t) * torch.randn_like(x)
                trajectory.append(x.clone())
        return torch.stack(trajectory, dim=1)

    def sample_ode(self, x0: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """ODE sampling using the antisymmetric drift (drift_fwd - drift_bwd) / 2."""
        model = self.get_ema_model()
        model.eval()
        n = num_steps if num_steps is not None else self.cfg.steps
        delta_t = self.T / n
        t_list = [torch.tensor([t], device=self.device)
                  for t in torch.linspace(0.0, self.T, n + 1)]
        ones = torch.ones(x0.shape[0], device=self.device)
        zeros = torch.zeros(x0.shape[0], device=self.device)

        x = x0.clone()
        trajectory = [x.clone()]
        with torch.no_grad():
            for t in t_list[:-1]:
                t_in = t.expand(x.shape[0], 1)
                drift_f = model(x, t_in, ones)
                drift_b = model(x, 1.0 - t_in, zeros)
                x = x + delta_t * 0.5 * (drift_f - drift_b)
                trajectory.append(x.clone())
        return torch.stack(trajectory, dim=1)

    def sample(self, x0: torch.Tensor, method: str = "ode",
               direction: str = "forward", num_steps: int | None = None) -> torch.Tensor:
        if method == "ode":
            return self.sample_ode(x0, num_steps=num_steps)
        return self.sample_sde(x0, direction=direction, num_steps=num_steps)

    # ------------------------------------------------------------------
    # WandB logging hooks (no-op if wandb_run is None)
    # ------------------------------------------------------------------
    def _log_pretrain(self, loss, loss_f, loss_b, v_f, v_b, epoch,
                      print_every, fixed_test_batch):
        if self.wandb_run is None:
            return
        self.wandb_run.log({
            "pretrain/loss": loss.item(),
            "pretrain/fwd_loss": loss_f.item(),
            "pretrain/bwd_loss": loss_b.item(),
            "pretrain/v_forward_norm": v_f.detach().norm(dim=-1).mean().item(),
            "pretrain/v_backward_norm": v_b.detach().norm(dim=-1).mean().item(),
            "epoch": epoch,
        }, step=self.global_step)
        if (self.global_step % print_every == 0
                and fixed_test_batch is not None
                and self.plot_fn is not None):
            fig = self.plot_fn(self, fixed_test_batch,
                               f"Pretrain Step {self.global_step}")
            self.wandb_run.log({"pretrain/samples": self.wandb_run.Image(fig)},
                               step=self.global_step)

    def _log_finetune(self, loss, loss_f, loss_b, v_f, v_b, lr_scheduler,
                      alpha, beta, lam, epoch, print_every, fixed_test_batch):
        if self.wandb_run is None:
            return
        self.wandb_run.log({
            "finetune/loss": loss.item(),
            "finetune/fwd_loss": loss_f.item(),
            "finetune/bwd_loss": loss_b.item(),
            "finetune/v_forward_norm": v_f.detach().norm(dim=-1).mean().item(),
            "finetune/v_backward_norm": v_b.detach().norm(dim=-1).mean().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "finetune/alpha": alpha.item(),
            "finetune/beta": beta.item(),
            "finetune/lambda": lam,
            "epoch": epoch,
        }, step=self.global_step)
        if (self.global_step % print_every == 0
                and fixed_test_batch is not None
                and self.plot_fn is not None):
            fig = self.plot_fn(self, fixed_test_batch,
                               f"Finetune Step {self.global_step}")
            self.wandb_run.log({"finetune/samples": self.wandb_run.Image(fig)},
                               step=self.global_step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, filepath, optimizer=None, lr_scheduler=None,
                        leap_scheduler=None, osc_scheduler=None, epoch: int = 0,
                        max_keep: int = 3) -> None:
        ckpt = {
            "model_state_dict": self.v_theta.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "loss_history": self.loss_history.__dict__,
            "global_step": self.global_step,
            "epoch": epoch,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if lr_scheduler is not None:
            ckpt["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        if leap_scheduler is not None:
            ckpt["leap_scheduler_state"] = {"step_idx": leap_scheduler.step_idx}
        if osc_scheduler is not None:
            ckpt["osc_scheduler_state"] = osc_scheduler.state_dict()

        torch.save(ckpt, filepath)
        print(f"Checkpoint saved: {filepath}")

        # Rolling deletion: keep only the last ``max_keep`` checkpoints.
        ckpt_dir = os.path.dirname(filepath)
        all_ckpts = sorted(
            (f for f in os.listdir(ckpt_dir) if f.endswith(".pt")),
            key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
        )
        while len(all_ckpts) > max_keep:
            os.remove(os.path.join(ckpt_dir, all_ckpts.pop(0)))

    def load_checkpoint(self, filepath, optimizer=None, lr_scheduler=None,
                        leap_scheduler=None, osc_scheduler=None) -> int:
        ckpt = torch.load(filepath, map_location=self.device)
        self.v_theta.load_state_dict(ckpt["model_state_dict"])
        self.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        for k, v in ckpt["loss_history"].items():
            setattr(self.loss_history, k, v)
        self.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        if leap_scheduler is not None and "leap_scheduler_state" in ckpt:
            leap_scheduler.step_idx = ckpt["leap_scheduler_state"]["step_idx"]
        if osc_scheduler is not None and "osc_scheduler_state" in ckpt:
            osc_scheduler.load_state_dict(ckpt["osc_scheduler_state"])

        print(f"Loaded {filepath} | epoch {start_epoch}, step {self.global_step}")
        return start_epoch + 1
