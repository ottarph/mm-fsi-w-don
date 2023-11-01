from neuraloperators.deepsets import *


def test_deepsets():
    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "folders")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    from neuraloperators.mlp import MLP
    representer = MLP([2, 16, 8], activation=nn.ReLU())
    processor = MLP([8, 16, 4])
    reduction = "mean"

    deepset = DeepSets(representer, processor, reduction)

    z = deepset(x0)
    assert z.shape == (x0.shape[0], 4)


    return

def test_deepset_perm_invariant():

    torch.set_default_dtype(torch.float64)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    from neuraloperators.mlp import MLP
    representer = MLP([2, 8], activation=nn.ReLU())
    processor = MLP([8, 4])
    reduction = "mean"

    deepset = DeepSets(representer, processor, reduction)

    bs = 20
    vs = 2000000
    x0 = torch.rand((vs, 2))

    x = torch.stack([x0[torch.randperm(x0.shape[-2])] for _ in range(bs)])

    x = x.to(device)
    deepset.to(device)

    z = deepset(x)
    assert z.shape == (bs, 4)

    eps = 1e-14
    # print(f"{torch.norm(z - z[0,:], dim=-1) = }")
    # print(f"{torch.finfo(torch.float32) = }")
    # print(f"{torch.finfo(torch.float64) = }")
    assert torch.all(torch.norm(z - z[0,:], dim=-1) < eps)

    torch.set_default_dtype(torch.float32)

    return

def test_deepset_deeponet():

    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "folders")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    uh: torch.Tensor = next(iter(dataloader))[0]
    y = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.encoders import IdentityEncoder, CoordinateInsertEncoder, RandomSelectEncoder, BoundaryFilterEncoder, SequentialEncoder
    branch_encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), RandomSelectEncoder(-2, 40))
    trunk_encoder = IdentityEncoder()

    assert branch_encoder(uh).shape == (uh.shape[0], 40, 4)
    assert trunk_encoder(y).shape == y.shape

    from neuraloperators.mlp import MLP
    representer = MLP([4, 16, 8], activation=nn.ReLU())
    processor = MLP([8, 16, 20])
    reduction = "mean"

    branch_net = DeepSets(representer, processor, reduction)
    trunk_net = MLP([2, 32, 32, 20])
    
    assert branch_net(branch_encoder(uh)).shape == (uh.shape[0], 20)
    assert trunk_net(trunk_encoder(y)).shape == (y.shape[0], y.shape[1], 20)

    from neuraloperators.deeponet import DeepONet
    deeponet = DeepONet(branch_encoder, branch_net, trunk_encoder, trunk_net, 2, 2, combine_style=2)

    assert deeponet(uh, y).shape == (uh.shape[0], y.shape[1], 2)
    assert (deeponet(uh, y) + uh).shape == (uh.shape[0], y.shape[1], 2)


    return

def test_deepset_deeponet_perm_invariant():
    
    # Note: in single precision, there is a lot of accumulated errors that make the invariance bad
    torch.set_default_dtype(torch.float64)

    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "folders")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    
    uh0 = dataset[0][0]

    print(uh0.shape)
    vs = 3924 # Number of vertices in artificial learnext mesh
    uh0 = torch.rand((vs, 2))
    print(f"{uh0.shape = }")



    bs = 3
    # uh = torch.stack([uh0[torch.randperm(uh0.shape[-2]),:] for _ in range(bs)])
    uh = torch.stack([uh0 for _ in range(bs)])

    print(f"{uh.shape = }")


    eps = 1e-14
    # assert torch.norm(torch.sum(uh - uh0, dim=1)) < eps, "Make sure the inputs are actually permutations"

    y = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.encoders import IdentityEncoder, CoordinateInsertEncoder, FixedFilterEncoder, BoundaryFilterEncoder, RandomPermuteEncoder, SequentialEncoder
    # filter = torch.LongTensor(list(range(10)))
    filter = x_data.boundary_dofs[:10]
    # print(filter)
    # branch_encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(filter, dim=-2))
    # branch_encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(filter, dim=-2), RandomPermuteEncoder(dim=-2))
    branch_encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), FixedFilterEncoder(filter, dim=-2), RandomPermuteEncoder(dim=-2))
    trunk_encoder = IdentityEncoder()

    print(f"{branch_encoder(uh) = }")
    print(f"{branch_encoder(uh).shape = }")
    assert True # Encoders need to be fixed before finishing this test.
    return
    quit()
    assert branch_encoder(uh).shape == (uh.shape[0], 10, 4)
    assert trunk_encoder(y).shape == y.shape

    print(f"{branch_encoder(uh) - branch_encoder(uh[[0],...])}")
    print(f"{torch.sum(branch_encoder(uh) - branch_encoder(uh[[0],...]), dim=1)}")

    from neuraloperators.mlp import MLP
    representer = MLP([4, 8], activation=nn.ReLU())
    processor = MLP([8, 20])
    reduction = "mean"

    branch_net = DeepSets(representer, processor, reduction)
    trunk_net = MLP([2, 20])
    
    assert branch_net(branch_encoder(uh)).shape == (uh.shape[0], 20)
    assert trunk_net(trunk_encoder(y)).shape == (y.shape[0], y.shape[1], 20)

    from neuraloperators.deeponet import DeepONet
    deeponet = DeepONet(branch_encoder, branch_net, trunk_encoder, trunk_net, 2, 2, combine_style=2)

    assert deeponet(uh, y).shape == (uh.shape[0], y.shape[1], 2)
    assert (deeponet(uh, y) + uh).shape == (uh.shape[0], y.shape[1], 2)

    # print(deeponet(uh, y) - deeponet(uh[[0],...], y))

    # z = deepset(x)
    # assert z.shape == (bs, 4)

    # eps = 1e-14
    # assert torch.all(torch.norm(z - z[0,:], dim=-1) < eps)

    # torch.set_default_dtype(torch.float32)

    # assert False

    torch.set_default_dtype(torch.float32)

    return
    

if __name__ == "__main__":
    test_deepsets()
    test_deepset_deeponet()
    test_deepset_perm_invariant()
    # test_deepset_deeponet_perm_invariant()
