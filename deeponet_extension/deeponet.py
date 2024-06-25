import torch
import torch.nn as nn

from deeponet_extension.encoders import Encoder
from functools import partial
from typing import Callable, Literal


class DeepONet(nn.Module):

    def __init__(self, branch_encoder: Encoder, branch: nn.Module, trunk_encoder: Encoder, trunk: nn.Module,
                       U_dim: int, V_dim: int,
                       final_bias: nn.Module | None = None, combine_style: Literal[2,3,4] = 2, 
                       sensors: torch.Tensor | None = None):
        """
            DeepONet implementation following presentation in https://arxiv.org/abs/2111.05512 [1], 
            in the form of vector output, even for one-dimensional output functions.

            Mappings from U to V. 
            ``U_dim`` and ``V_dim`` are the dimensionality of the input and output functions. 
            These are required arguments since ``V_dim`` is required for combining ``b_k`` and 
            ``t_k`` correctly.

            ``combine_style: Literal[1,2,3,4])`` refers to the way hidden features are combined 
            to create vector outputs, see [1, sec. 3.1.6].

            Normal forward call assumes point evaluated input function `u`, with evaluations in the 
            sensor locations, but does nothing to ensure this. If a callable function is to be used,
            use the ``.callable_forward()``-method along with chosen sensors. These can be stored as
            a property of the DeepONet with the ``sensors`` keyword argument.
        """
        super().__init__()

        self.branch_encoder = branch_encoder
        self.branch = branch
        self.trunk_encoder = trunk_encoder
        self.trunk = trunk

        self.U_dim = U_dim
        self.V_dim = V_dim

        self.final_bias = final_bias
        self.combine_style = combine_style

        if isinstance(sensors, torch.Tensor):
            self.register_buffer("sensors", sensors)
        else:
            self.sensors = sensors
        self.sensors: torch.Tensor | None

        return
    
    def combine_2(self, bk: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        
        assert bk.shape[-1] == tk.shape[-1]
        assert bk.shape[-1] % self.V_dim == 0

        if len(bk.shape) < len(tk.shape):
            bk = torch.unsqueeze(bk, dim=-2)

        N = bk.shape[-1] // self.V_dim

        contract = partial(torch.einsum, "...i,...i->...")
        vh = torch.stack(list(map(contract, torch.split(bk, N, -1), torch.split(tk, N, -1))), dim=-1)

        return vh
    
    def combine_3(self, bk: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:

        assert bk.shape[-1] == tk.shape[-1] * self.V_dim
        assert tk.shape[-1] % self.V_dim == 0

        if len(bk.shape) < len(tk.shape):
            bk = torch.unsqueeze(bk, dim=-2)

        N = bk.shape[-1] // self.V_dim

        contract = partial(torch.einsum, "...i,...i->...")
        vh = torch.stack(list(map(contract, torch.split(bk, N, -1), [tk] * N)), dim=-1)

        return vh
    
    def combine_4(self, bk: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        
        assert bk.shape[-1] * self.V_dim == tk.shape[-1]
        assert bk.shape[-1] % self.V_dim == 0

        if len(bk.shape) < len(tk.shape):
            bk = torch.unsqueeze(bk, dim=-2)

        N = tk.shape[-1] // self.V_dim

        contract = partial(torch.einsum, "...i,...i->...")
        vh = torch.stack(list(map(contract, [bk] * N, torch.split(tk, N, -1))), dim=-1)

        return vh
        
    
    def combine(self, bk: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:

        out = {2: self.combine_2, 
               3: self.combine_3, 
               4: self.combine_4}[self.combine_style](bk, tk)

        return out
    
    def forward(self, uh: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bk = self.branch(self.branch_encoder(uh))
        tk = self.trunk(self.trunk_encoder(y))

        vh = self.combine(bk, tk)

        if self.final_bias is not None:
            vh = vh + self.final_bias(self.trunk_encoder(y))

        return vh
    
    def __call__(self, uh: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(u, y) be tensor
        return super().__call__(uh, y)
    
    def callable_forward(self, u: Callable[[torch.Tensor], torch.Tensor], sensors: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        uh = u(sensors)
        return self(uh, y)



