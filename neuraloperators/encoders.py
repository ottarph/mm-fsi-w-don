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

class FlattenEncoder(Encoder):
    def __init__(self, start_dim: int):
        """

        """
        super().__init__()

        self.register_buffer("start_dim", torch.tensor(start_dim, dtype=torch.long))
        self.start_dim: torch.LongTensor

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.flatten(x, start_dim=self.start_dim)
    

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


class EncoderBuilder:

    def SequentialEncoder(encoder_dicts: list[dict]) -> SequentialEncoder:
        return SequentialEncoder(*(build_encoder(enc_dict) for enc_dict in encoder_dicts))
    
    def CoordinateInsertEncoder(coord_enc_dict: dict) -> CoordinateInsertEncoder:
        """
            ```
                coord_enc_dict = {"dataset_path": path/to/dataset_dir, "side": "input" / "output"}
            ```
        """
        from dataset.dataset import load_MeshData
        x_mesh_data, y_mesh_data = load_MeshData(coord_enc_dict["dataset_path"], style="XDMF")

        if coord_enc_dict["side"] == "input":
            mesh_data = x_mesh_data
        elif coord_enc_dict["side"] == "output":
            mesh_data = y_mesh_data
        else:
            raise ValueError()

        return CoordinateInsertEncoder(mesh_data)
    
    def BoundaryFilterEncoder(bound_filt_dict: dict) -> BoundaryFilterEncoder:
        """
            ```
                coord_enc_dict = {"dataset_path": path/to/dataset_dir, "side": "input" / "output"}
            ```
        """
        from dataset.dataset import load_MeshData
        x_mesh_data, y_mesh_data = load_MeshData(bound_filt_dict["dataset_path"], style="XDMF")

        if bound_filt_dict["side"] == "input":
            mesh_data = x_mesh_data
        elif bound_filt_dict["side"] == "output":
            mesh_data = y_mesh_data
        else:
            raise ValueError()

        return BoundaryFilterEncoder(mesh_data)
    
    def RandomPermuteEncoder(rand_perm_dict: dict) -> RandomPermuteEncoder:
        assert isinstance(rand_perm_dict["dim"], int) and rand_perm_dict["dim"] >= 0
        return RandomPermuteEncoder(rand_perm_dict["dim"])
    
    def RandomSelectEncoder(rand_sel_dict: dict) -> RandomSelectEncoder:
        assert isinstance(rand_sel_dict["dim"], int) and rand_sel_dict["dim"] >= 0
        assert isinstance(rand_sel_dict["num_inds"], int) and rand_sel_dict["num_inds"] > 0
        return RandomPermuteEncoder(rand_sel_dict["dim"])


def build_encoder(encoder_dict: dict) -> Encoder:

    assert len(encoder_dict.keys()) == 1
    key = next(iter(encoder_dict.keys()))
    val = encoder_dict[key]

    encoder: nn.Module = getattr(EncoderBuilder, key)(val)

    return encoder

