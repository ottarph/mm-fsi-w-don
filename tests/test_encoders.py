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

    return

def test_filter_encoders():

    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    rand_perm_enc = RandomPermuteEncoder(-2)

    assert rand_perm_enc(x0).shape == x0.shape
    assert torch.norm(x0 - rand_perm_enc(x0)) != 0

    rand_select_enc = RandomSelectEncoder(-2, 20)

    assert rand_select_enc(x0).shape == (x0.shape[0], rand_select_enc.num_inds, x0.shape[-1])
    assert torch.norm(x0[:,:rand_select_enc.num_inds,:] - rand_select_enc(x0)) != 0

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
    rand_select_encoder = RandomSelectEncoder(-2, 20)

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
    rand_select_encoder = RandomSelectEncoder(-2, 20) # Select 20 vertices at random, (batch, num_bound_vert, 4) -> (batch, 20, 4)
    flat_encoder = FlattenEncoder(1) # Flatten tensor starting at second dimension, (batch, 20, 4) -> (batch, 20*4=80)
    encoder = SequentialEncoder(coordinate_insert_encoder, boundary_filter_encoder, rand_select_encoder, flat_encoder)

    assert encoder(x0).shape == (x0.shape[0], 20*4)

    return


if __name__ == "__main__":
    test_coordinate_insert()
    test_filter_encoders()
    test_combined_encoders()
    test_flatten_encoder()
