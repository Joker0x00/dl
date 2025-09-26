import os
import logging
from typing import Dict, List

import torch

from src.utils import save_checkpoint, epoch_checkpoint_path, best_checkpoint_path

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, metrics: Dict[str, callable], device: torch.device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device
        self.cfg = cfg
        self.best_score = None

    def _step(self, batch, train=True):
        batch = [b.to(self.device) for b in batch]
        x = batch[:3]
        y = batch[3]
        logits = self.model(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2])
        loss = self.loss_fn(logits, y)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.train.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss.item(), logits.detach(), y.detach()

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_loss, n = 0.0, 0
        agg = {k: 0.0 for k in self.metrics}
        for batch in loader:
            batch = [b.to(self.device) for b in batch]
            x = batch[:3]
            y = batch[3]
            logits = self.model(**x)
            loss = self.loss_fn(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
            for name, fn in self.metrics.items():
                agg[name] += fn(logits, y) * bs
        self.model.train()
        return {"loss": total_loss / max(n,1), **{k: v / max(n,1) for k,v in agg.items()}}

    def fit(self, train_loader, val_loader):
        os.makedirs(self.cfg.train.save_dir, exist_ok=True)
        monitor = self.cfg.train.monitor
        mode = self.cfg.train.monitor_mode
        greater_is_better = (mode.lower() == "max")

        history: List[dict] = []
        patience = self.cfg.train.early_stop_patience
        strikes = 0

        for epoch in range(1, self.cfg.train.epochs + 1):
            logging.info(f"Epoch {epoch}/{self.cfg.train.epochs}")
            self.model.train()
            running = {"loss": 0.0}
            for name in self.metrics:
                running[name] = 0.0
            n = 0

            for batch in train_loader:
                loss, logits, y = self._step(batch, train=True)
                bs = y.size(0)
                running["loss"] += loss * bs
                for name, fn in self.metrics.items():
                    running[name] += fn(logits, y) * bs
                n += bs

            train_logs = {k: v / max(n,1) for k, v in running.items()}
            val_logs = self._evaluate(val_loader)

            logs = {**{f"train/{k}": v for k, v in train_logs.items()}, **{f"val/{k}": v for k, v in val_logs.items()}}
            history.append({"epoch": epoch, **logs})
            logging.info("Metrics: %s", {k: round(v, 5) for k, v in logs.items()})

            # 保存每轮
            if self.cfg.train.save_every_epoch:
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
                    "cfg": self.cfg,
                    "history": history,
                }, epoch_checkpoint_path(self.cfg.train.save_dir, epoch))

            # 保存最好
            score = self._read_monitor(logs, monitor)
            if self.best_score is None or (greater_is_better and score > self.best_score) or (not greater_is_better and score < self.best_score):
                self.best_score = score
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
                    "cfg": self.cfg,
                    "history": history,
                }, best_checkpoint_path(self.cfg.train.save_dir))
                logging.info("New best %.5f on %s, saved to %s", score, monitor, best_checkpoint_path(self.cfg.train.save_dir))
                strikes = 0
            else:
                strikes += 1
                logging.info("No improvement (%d/%s)", strikes, patience)
                if patience and strikes >= patience:
                    logging.info("Early stopping.")
                    break

        return history

    @staticmethod
    def _read_monitor(logs: Dict[str, float], key: str) -> float:
        assert key in logs, f"monitor key {key} not found in logs"
        return float(logs[key])