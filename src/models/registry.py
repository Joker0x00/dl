import importlib
import torch.nn as nn
from typing import Optional

from .bert_peft import BertPEFT

_BUILTIN = {
    "bert_peft": BertPEFT,
}

try:
    import torchvision.models as tvm
    _BUILTIN.update({
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
    })
except Exception:
    pass


def build_model(name: str, import_path: Optional[str] = None, class_name: Optional[str] = None, **kwargs) -> nn.Module:
    name = (name or "").lower()
    if name == "custom":
        assert import_path and class_name, "custom 模型需要 import_path 与 class_name"
        module = importlib.import_module(import_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    if name in _BUILTIN:
        return _BUILTIN[name](**kwargs)
    raise ValueError(f"Unknown model name: {name}")