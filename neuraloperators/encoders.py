import torch
import torch.nn as nn

from dataset.dataset import MeshData


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()


class SequentialEncoder(Encoder):
    def __init__(self, *args):
        super().__init__()

        self.encoders = nn.Sequential(*args)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoders(x)

    
class TensorPrependEncoder(Encoder):
    """ Inserts `tensor` before `x` in the last dimension. """
    
    def __init__(self, tensor: torch.Tensor):
        super().__init__()

        self.register_buffer("tensor", tensor)
        self.tensor: torch.Tensor
        self.tensor.requires_grad_(False)
    
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.cat((self.tensor.expand((x.shape[0], -1, -1)), x), dim=-1)


class CoordinateInsertEncoder(TensorPrependEncoder):
    
    def __init__(self, mesh_data: MeshData):
        
        coordinates = mesh_data.dof_coordinates.to(torch.get_default_dtype())
        super().__init__(coordinates)
        self.coordinates: torch.Tensor = self.tensor

        return


class FilterEncoder(Encoder):

    def __init__(self, dim: int):
        super().__init__()

        self.register_buffer("dim", torch.tensor(dim, dtype=torch.long))
        self.dim: torch.LongTensor

        return

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return torch.index_select(x, self.dim, self.filter(x))
    

class FixedFilterEncoder(FilterEncoder):

    def __init__(self, filter: torch.LongTensor, dim: int):
        super().__init__(dim=dim)
        self.register_buffer("filter_tensor", filter)
        self.filter_tensor: torch.LongTensor

        return
    
    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return self.filter_tensor
    

class BoundaryFilterEncoder(FixedFilterEncoder):

    def __init__(self, mesh_data: MeshData):
        indices = mesh_data.boundary_dofs

        super().__init__(filter=indices, dim=-2)
        
        return


class RandomPermuteEncoder(FilterEncoder):

    def __init__(self, dim: int):
        super().__init__(dim=dim)

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.randperm(x.shape[self.dim], device=x.device)
    
    
class RandomSelectEncoder(FilterEncoder):

    def __init__(self, dim: int, num_inds: int):
        super().__init__(dim=dim)
        self.register_buffer("num_inds", torch.tensor(num_inds, dtype=torch.long))
        self.num_inds: torch.LongTensor

        return

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.randperm(x.shape[self.dim], device=x.device)[:self.num_inds]

