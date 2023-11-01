import torch
import torch.nn as nn

from typing import Literal

class DeepSets(nn.Module):

    def __init__(self, representer: nn.Module, processor: nn.Module, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()

        self.representer = representer
        self.processor = processor
        self.reduction = reduction
        self.reduction_call = {"mean": self.mean_reduction, "sum": self.sum_reduction}[reduction]

        return
    
    def mean_reduction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=-2)
    
    def sum_reduction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        phi = self.representer(x)
        reduced = self.reduction_call(phi)
        output = self.processor(reduced)

        return output
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(x) be tensor
        return super().__call__(x)
