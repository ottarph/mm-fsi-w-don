import torch
import torch.nn as nn

from typing import Callable, Sequence

class MLP(nn.Module):

    def __init__(self, widths: Sequence[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(x) be tensor
        return super().__call__(x)

    
__all__ = ["MLP"]
