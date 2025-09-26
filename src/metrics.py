import torch
from typing import Optional
from src.utils import register

@register("metric", "accuracy")
@torch.no_grad()
def accuracy(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> float:
    """适用于多分类：pred 为 logits 或概率 (N, C)"""
    if pred.ndim > 1:
        pred_label = pred.argmax(dim=1)
    else:
        pred_label = (pred > 0.5).long()
    return (pred_label == target).float().mean().item()

@register("metric", "f1")
@torch.no_grad()
def f1_score(pred: torch.Tensor, target: torch.Tensor, average: str = "macro", **kwargs) -> float:
    """轻量实现：支持二分类/多分类。"""
    if pred.ndim > 1:
        pred_label = pred.argmax(dim=1)
        num_classes = int(target.max().item() + 1)
    else:
        pred_label = (pred > 0.5).long()
        num_classes = 2

    f1s = []
    for c in range(num_classes):
        tp = ((pred_label == c) & (target == c)).sum().item()
        fp = ((pred_label == c) & (target != c)).sum().item()
        fn = ((pred_label != c) & (target == c)).sum().item()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    if average == "macro":
        return float(sum(f1s) / len(f1s))
    elif average == "micro":
        # micro F1 等于整体精确率/召回率的调和
        tp = (pred_label == target).sum().item()
        precision = recall = tp / len(target)
        return float(2 * precision * recall / (precision + recall + 1e-12))
    else:
        raise ValueError("average must be 'macro' or 'micro'")