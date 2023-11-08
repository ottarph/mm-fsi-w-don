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


class SplitAdditive(nn.Module):

    def __init__(self, module_1: nn.Module, module_2: nn.Module, length_1: int, length_2: int):
        super().__init__()

        self.module_1 = module_1
        self.module_2 = module_2

        self.length_1 = length_1
        self.length_2 = length_2

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module_1(x[...,:self.length_1]) + self.module_2(x[...,self.length_1:])

__all__ = ["MLP", "SplitAdditive"]
