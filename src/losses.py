import torch.nn as nn
from src.utils import register

# 常用损失可以直接映射
_BUILTIN = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
}

for k, v in _BUILTIN.items():
    register("loss", k)(v)