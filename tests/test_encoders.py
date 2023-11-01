from neuraloperators.encoders import *
from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

def test_coordinate_insert():

    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data)
    A = nn.Linear(4, 3)
    
    x0, _ = next(iter(dataloader))
    assert A(coordinate_insert_encoder(x0)).shape == (x0.shape[0], x0.shape[1], A.out_features)
    assert torch.max(coordinate_insert_encoder(x0)[0,:,0]) - 2.5 < torch.finfo(torch.get_default_dtype()).eps * 10

    # print(coordinate_insert_encoder(x0))

    return

def test_random_permute_encoder():

    # torch.set_default_dtype(torch.float64)

    vertex_size = 2
    num_verts = 5
    x0 = torch.rand((num_verts, vertex_size))
    bs = 4
    x1 = torch.stack([x0] * bs)
    x2 = torch.stack([x1, -x1], dim=0)

    class Permuter(nn.Module):
        def __init__(self, dim: int, unit_shape_length: int):
            super().__init__()
            self.register_buffer("dim", torch.tensor(dim, dtype=torch.long))
            self.dim: torch.LongTensor

            self.register_buffer("unit_shape_length", torch.tensor(unit_shape_length, dtype=torch.long))
            self.unit_shape_length: torch.LongTensor
            
            self.permute_batch = torch.vmap(self.permute_single, randomness="different")
            self.permute_batch_double = torch.vmap(self.permute_batch, randomness="different")
            return
        
        def permute_single(self, x: torch.Tensor) -> torch.Tensor:
            filter = torch.randperm(x.shape[self.dim], device=x.device)
            out = torch.index_select(x, dim=self.dim, index=filter)
            return out
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if len(x.shape) == self.unit_shape_length:
                out = self.permute_single(x)
            elif len(x.shape) == self.unit_shape_length + 1:
                out = self.permute_batch(x)
            elif len(x.shape) == self.unit_shape_length + 2:
                out = self.permute_batch_double(x)
            else:
                raise ValueError
            return out
  
    # permuter = Permuter(-2, 2)
    permuter = RandomPermuteEncoder(-2, 2)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    permuter.to(device)
    x0 = x0.to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)

    eps = torch.finfo(torch.get_default_dtype()).eps

    # Test no batch:
    # print(eps)
    # print(dir(torch.finfo(torch.float32)))
    # print(torch.sum(permuter(x0) - x0, dim=-2))
    # print(torch.norm(torch.sum(permuter(x0) - x0, dim=-2)))
    # print(torch.norm(permuter(x0) - x0))
    assert torch.norm(torch.sum(permuter(x0) - x0, dim=-2)) < 10*eps # Assert produced tensor is actually a permutation of x0
    assert torch.norm(permuter(x0) - x0) > 100*eps # Assert produced tensor is changed
    # print(100*eps)

    # Test single batch dimension
    # print(torch.sum(permuter(x1) - x0, dim=-2))
    # print(torch.norm(torch.sum(permuter(x1) - x0, dim=-2)))
    # print(permuter(x1).shape)
    # print(permuter(x1) - permuter(x1)[0,...])
    assert torch.norm(torch.sum(permuter(x1) - x0, dim=-2)) < 10*eps # Assert produced tensor is actually a permutation of x0
    assert torch.norm(permuter(x1) - x1) > 100*eps # Assert produced tensor is changed

    # Test double batch dimension
    # print(torch.sum(permuter(x2)[0,...] - x0, dim=-2))
    # print(torch.sum(permuter(x2)[1,...] + x0, dim=-2))
    # print(torch.norm(torch.sum(permuter(x2)[0,...] - x0, dim=-2)))
    # print(torch.norm(torch.sum(permuter(x2)[1,...] + x0, dim=-2)))
    assert torch.norm(torch.sum(permuter(x2)[0,...] - x0, dim=-2)) < 10*eps # Assert produced tensor is actually a permutation of x0
    assert torch.norm(torch.sum(permuter(x2)[1,...] + x0, dim=-2)) < 10*eps # Assert produced tensor is actually a permutation of -x0
    assert torch.norm(permuter(x2) - x2) > 100*eps # Assert produced tensor is changed


    # print(permuter(x2))
    # print(permuter(x1))
    # print(permuter(x0))
    # print(x0)

    # torch.set_default_dtype(torch.float32)

    return

def test_filter_encoders():

    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    rand_perm_enc = RandomPermuteEncoder(-2, 2)

    assert rand_perm_enc(x0).shape == x0.shape
    assert torch.norm(x0 - rand_perm_enc(x0)) != 0

    rand_select_enc = RandomSelectEncoder(-2, 2, 20)

    assert rand_select_enc(x0).shape == (x0.shape[0], rand_select_enc.num_inds, x0.shape[-1])
    assert torch.norm(x0[:,:rand_select_enc.num_inds,:] - rand_select_enc(x0)) != 0

    x1 = torch.stack([x0[0,...], x0[0,...]], dim=0)
    rsx1 = rand_select_enc(x1)
    assert torch.norm(rsx1[0,...] - rsx1[1,...]) > torch.finfo(torch.get_default_dtype()).eps * 100
    
    return

def test_combined_encoders():

    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data)
    boundary_filter_encoder = BoundaryFilterEncoder(x_data)
    rand_select_encoder = RandomSelectEncoder(-2, 2, 20)

    encoder = SequentialEncoder(coordinate_insert_encoder, boundary_filter_encoder, rand_select_encoder)

    assert isinstance(encoder, Encoder)
    assert encoder(x0).shape == (x0.shape[0], rand_select_encoder.num_inds, 4)
    assert torch.norm(x0[:,:rand_select_encoder.num_inds,:] - encoder(x0)[:, :, 2:]) != 0

    A = nn.Linear(4, 3)
    assert A(encoder(x0)).shape == (x0.shape[0], rand_select_encoder.num_inds, A.out_features) 

    return


def test_flatten_encoder():

    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    flat_encoder = FlattenEncoder(1)
    assert flat_encoder(x0).shape == (x0.shape[0], x0.shape[1]*x0.shape[2])

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data) # Insert coordinates, (batch, num_vert, 2) -> (batch, num_vert, 4)
    boundary_filter_encoder = BoundaryFilterEncoder(x_data) # Select only boundary vertices, (batch, num_vert, 4) -> (batch, num_bound_vert, 4)
    rand_select_encoder = RandomSelectEncoder(-2, 2, 20) # Select 20 vertices at random, (batch, num_bound_vert, 4) -> (batch, 20, 4)
    flat_encoder = FlattenEncoder(1) # Flatten tensor starting at second dimension, (batch, 20, 4) -> (batch, 20*4=80)
    encoder = SequentialEncoder(coordinate_insert_encoder, boundary_filter_encoder, rand_select_encoder, flat_encoder)

    assert encoder(x0).shape == (x0.shape[0], 20*4)

    return


if __name__ == "__main__":
    test_coordinate_insert()
    test_random_permute_encoder()
    test_filter_encoders()
    test_combined_encoders()
    test_flatten_encoder()
