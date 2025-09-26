import os, json, random
import numpy as np
import torch
from typing import Dict, Callable, Any

REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "loss": {},
    "metric": {},
}

def register(kind: str, name: str):
    def deco(fn):
        REGISTRY[kind][name] = fn
        return fn
    return deco

# --------- 通用 ---------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


def best_checkpoint_path(save_dir: str) -> str:
    return os.path.join(save_dir, "best.pt")


def epoch_checkpoint_path(save_dir: str, epoch: int) -> str:
    return os.path.join(save_dir, f"epoch_{epoch:03d}.pt")


def write_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)