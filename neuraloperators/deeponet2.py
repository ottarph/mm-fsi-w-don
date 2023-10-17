import torch
import torch.nn as nn

from neuraloperators.encoders import Encoder, FunctionRepresentation

from typing import Callable


class DeepONet(nn.Module):

    def __init__(self, branch_encoder: Encoder, branch: nn.Module, trunk_encoder: Encoder, trunk: nn.Module,
                       final_bias: nn.Module | None):
        super().__init__()

        self.branch_encoder = branch_encoder
        self.branch = branch
        self.trunk_encoder = trunk_encoder
        self.trunk = trunk

        self.final_bias = final_bias

        return
    
    def forward(self, uh: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bk = self.branch(self.branch_encoder(uh))
        tk = self.trunk(self.trunk_encoder(y))

        z = torch.einsum("...k,...k->...", bk, tk)

        if self.final_bias:
            z = z + self.final_bias(self.trunk_encoder(y))

        return
    
    def callable_forward(self, u: Callable[[torch.Tensor], torch.Tensor], sensors: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        uh = u(sensors)
        return self(uh, y)



