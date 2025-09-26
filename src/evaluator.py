import torch
from typing import Dict

class Evaluator:
    def __init__(self, model, loss_fn, metrics: Dict[str, callable], device: torch.device):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, n = 0.0, 0
        agg = {k: 0.0 for k in self.metrics}
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
            for name, fn in self.metrics.items():
                agg[name] += fn(logits, y) * bs
        return {"loss": total_loss / max(n,1), **{k: v / max(n,1) for k,v in agg.items()}}