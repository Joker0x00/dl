import logging
import torch
from src.utils import load_checkpoint


def resume_from(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = load_checkpoint(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])  # 严格加载
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt.get("epoch", 0)
    logging.info("Resumed from %s at epoch %d", path, start_epoch)
    return start_epoch, ckpt