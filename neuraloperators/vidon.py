import torch
import torch.nn as nn
import numpy as np

from neuraloperators.networks import MLP

from typing import Callable


class VIDONMHAHead(nn.Module):

    def __init__(self, d_enc: int, out_size: int, 
                 weight_hidden_size: int, weight_num_layers: int, value_hidden_size: int, value_num_layers: int, 
                 weight_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(), value_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.d_enc = d_enc
        self.out_size = out_size

        weight_widths = [d_enc] + [weight_hidden_size] * (weight_num_layers - 2) + [1]
        self.weight_mlp = MLP(weight_widths, weight_activation)

        value_widths = [d_enc] + [value_hidden_size] * (value_num_layers - 2) + [out_size]
        self.value_mlp = MLP(value_widths, value_activation)

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        values = self.value_mlp(x)
        pre_weights = self.weight_mlp(x)
        weights = nn.functional.softmax(pre_weights / np.sqrt(self.d_enc), dim=-2)
        out = torch.einsum("...ki,...ki->...i", values, weights)

        return out
    

class VIDONMHA(nn.Module):

    def __init__(self, d_enc: int, out_size: int, num_heads: int, 
                 weight_hidden_size: int, weight_num_layers: int, value_hidden_size: int, value_num_layers: int, 
                 weight_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(), value_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.heads = nn.ModuleList([VIDONMHAHead(d_enc, out_size, 
                                    weight_hidden_size, weight_num_layers, 
                                    value_hidden_size, value_num_layers,
                                    weight_activation=weight_activation, value_activation=value_activation)] * num_heads)
        self.d_enc = d_enc
        self.out_size = out_size
        self.num_heads = num_heads

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out


from neuraloperators.networks import SplitAdditive

class VIDON(nn.Module):

    # def __init__(self, coord_dim: int, value_dim: int, d_enc: int, num_heads: int, num_branch_features: int,
    #              )

    def __init__(self, split_encoder: SplitAdditive, multiheadattention: VIDONMHA, processor: MLP):
        super().__init__()

        self.split_encoder = split_encoder
        self.multiheadattention = multiheadattention
        self.processor = processor

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.split_encoder(x)
        out = self.multiheadattention(out)
        out = self.processor(out)

        return out


