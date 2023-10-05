import torch
import torch.nn as nn


class DataInformedLoss(nn.modules.loss._Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RelativeMSELoss(DataInformedLoss):

    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.eps = eps

        return

    def forward(self, input: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:


        return torch.sum( ( (target - pred)**2 / ( (target - input)**2 + self.eps ) ) )
    