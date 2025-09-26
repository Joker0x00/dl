# src/models/mlp.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_classes: int):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(0.1)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)
    def forward(self, x):
        return self.head(self.backbone(x))