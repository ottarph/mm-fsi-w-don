import torch
import torch.nn as nn

from typing import Callable, Sequence

class MLP(nn.Module):

    def __init__(self, widths: Sequence[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        self.layers = nn.ModuleList([nn.Linear(w1, w2) for (w1, w2) in zip(widths[:-1], widths[1:])])

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        return y
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(x) be tensor
        return super().__call__(x)

