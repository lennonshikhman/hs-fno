from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hsno.utils.seed import seed_worker, torch_generator
from tqdm import tqdm
from .losses import data_loss, rollout_loss, semiflow_loss


class Trainer:
    def __init__(self, model, train_ds, val_ds, cfg, device):
        self.model = model.to(device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.device = device
        train_cfg = cfg["training"]
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

    def _move(self, b):
        return {k: v.to(self.device) for k, v in b.items()}

    def fit(self, ckpt_path=None):
        train_cfg = self.cfg["training"]
        hist = []
        bs = train_cfg["batch_size"]
        epochs = int(train_cfg["epochs"])
        patience = train_cfg.get("patience")
        patience = None if patience is None else int(patience)
        min_delta = float(train_cfg.get("min_delta", 0.0))
        grad_clip = train_cfg.get("grad_clip_norm")
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        seed = int(self.cfg.get("seed", 0))
        loader_kwargs = dict(worker_init_fn=seed_worker, generator=torch_generator(seed), num_workers=int(train_cfg.get("num_workers", 0)))

        for ep in range(epochs):
            self.model.train()
            losses = []
            for b in tqdm(DataLoader(self.train_ds, batch_size=bs, shuffle=True, **loader_kwargs), desc=f"epoch {ep + 1}/{epochs}", leave=False):
                b = self._move(b)
                pred = self.model(b["history"], b["cond"], b["static"])
                loss = data_loss(pred, b["target_history"])
                if train_cfg.get("rollout_weight", 0) > 0:
                    loss = loss + train_cfg["rollout_weight"] * rollout_loss(self.model, b, train_cfg.get("rollout_steps", 2))
                if train_cfg.get("semiflow_weight", 0) > 0:
                    loss = loss + train_cfg["semiflow_weight"] * semiflow_loss(self.model, b)
                self.opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                self.opt.step()
                losses.append(loss.item())

            val = self.evaluate_loss()
            train_loss = float(sum(losses) / max(1, len(losses)))
            improved = val < best_val - min_delta
            if improved:
                best_val = val
                bad_epochs = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
            hist.append(
                {
                    "epoch": ep + 1,
                    "train_loss": train_loss,
                    "val_loss": val,
                    "best_val_loss": float(best_val),
                    "bad_epochs": bad_epochs,
                    "early_stopped": False,
                }
            )
            if patience is not None and bad_epochs >= patience:
                hist[-1]["early_stopped"] = True
                break

        if best_state is not None and train_cfg.get("restore_best", True):
            self.model.load_state_dict(best_state)
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path)
        return hist

    def evaluate_loss(self):
        self.model.eval()
        vals = []
        with torch.no_grad():
            for b in DataLoader(self.val_ds, batch_size=self.cfg["training"]["batch_size"], worker_init_fn=seed_worker, generator=torch_generator(int(self.cfg.get("seed", 0))), num_workers=int(self.cfg["training"].get("num_workers", 0))):
                b = self._move(b)
                vals.append(data_loss(self.model(b["history"], b["cond"], b["static"]), b["target_history"]).item())
        return float(sum(vals) / max(1, len(vals)))
