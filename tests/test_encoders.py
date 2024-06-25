from neuraloperators.encoders import *
from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

def test_coordinate_insert():

    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data)
    A = nn.Linear(4, 3)
    
    x0, _ = next(iter(dataloader))
    assert A(coordinate_insert_encoder(x0)).shape == (x0.shape[0], x0.shape[1], A.out_features)
    assert torch.max(coordinate_insert_encoder(x0)[0,:,0]) - 2.5 < torch.finfo(torch.get_default_dtype()).eps * 10

    # print(coordinate_insert_encoder(x0))

    return


def test_inner_boundary_filter():

    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), InnerBoundaryFilterEncoder(x_data))

    assert encoder(x0).shape == (x0.shape[0], 206, 4)

    return

def test_combined_encoders():

    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data)
    boundary_filter_encoder = BoundaryFilterEncoder(x_data)

    encoder = SequentialEncoder(coordinate_insert_encoder, boundary_filter_encoder)

    assert isinstance(encoder, Encoder)
    assert encoder(x0).shape == (x0.shape[0], boundary_filter_encoder.filter_tensor.shape[0], 4)

    A = nn.Linear(4, 3)
    assert A(encoder(x0)).shape == (x0.shape[0], boundary_filter_encoder.filter_tensor.shape[0], A.out_features) 

    return


def test_flatten_encoder():

    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    flat_encoder = FlattenEncoder(1)
    assert flat_encoder(x0).shape == (x0.shape[0], x0.shape[1]*x0.shape[2])

    coordinate_insert_encoder = CoordinateInsertEncoder(x_data) # Insert coordinates, (batch, num_vert, 2) -> (batch, num_vert, 4)
    boundary_filter_encoder = BoundaryFilterEncoder(x_data) # Select only boundary vertices, (batch, num_vert, 4) -> (batch, num_bound_vert, 4)
    flat_encoder = FlattenEncoder(1) # Flatten tensor starting at second dimension, (batch, num_bound_vert, 4) -> (batch, num_bound_vert*4)
    encoder = SequentialEncoder(coordinate_insert_encoder, boundary_filter_encoder, flat_encoder)

    assert encoder(x0).shape == (x0.shape[0], boundary_filter_encoder.filter_tensor.shape[0]*4)

    return


if __name__ == "__main__":
    test_coordinate_insert()
    test_inner_boundary_filter()
    test_combined_encoders()
    test_flatten_encoder()
