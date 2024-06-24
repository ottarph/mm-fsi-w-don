from neuraloperators.vidon import *

from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

from neuraloperators.networks import MLP, SplitAdditive
from neuraloperators.encoders import SequentialEncoder, CoordinateInsertEncoder, BoundaryFilterEncoder, FixedFilterEncoder, SplitAdditiveEncoder

def test_split_additive():

    split_add_model = SplitAdditive(MLP([2, 64]), MLP([2, 64]), 2, 2)

    x = torch.rand((8, 60, 4))
    assert split_add_model(x).shape == (8, 60, 64)

    x = torch.rand((4))
    x1 = torch.cat([x[:2], x[:2]], dim=0)
    x2 = torch.cat([x[:2], x[2:]], dim=0)
    x3 = torch.cat([x[2:], x[:2]], dim=0)
    x4 = torch.cat([x[2:], x[2:]], dim=0)
    y1 = split_add_model(x1)
    y2 = split_add_model(x2)
    y3 = split_add_model(x3)
    y4 = split_add_model(x4)

    # Check split-additivity. If correctly implemented, then
    #             y1 + y4 = sa_1(x[:2]) + sa_1(x[2:]) + sa_2(x[:2]) + sa_2(x[2:])
    #             y2 + y3 = sa_1(x[:2]) + sa_1(x[2:]) + sa_2(x[2:]) + sa_2(x[:2])
    # y1 + y4 - (y2 + y3) = 0
    eps = torch.finfo(torch.get_default_dtype()).eps
    assert torch.norm( y1 + y4 - y2 - y3) < eps * 64

    return

def test_VIDONMHAHead():
    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
        x_transform=ToDType("default"),
        y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    splenc_1 = MLP([2, 64], nn.ReLU())
    splenc_2 = MLP([2, 64], nn.ReLU())
    encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(torch.LongTensor(list(range(6))), -2, 2), SplitAdditiveEncoder(splenc_1, splenc_2, 2, 2))

    vmhahead = VIDONMHAHead(64, 32, 256, 4, 256, 4)
    out = vmhahead(encoder(x0))
    assert out.shape == (x0.shape[0], 32)

    return



def test_VIDONMHA():
    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
        x_transform=ToDType("default"),
        y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    splenc_1 = MLP([2, 64], nn.ReLU())
    splenc_2 = MLP([2, 64], nn.ReLU())
    encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(torch.LongTensor(list(range(6))), -2, 2), SplitAdditiveEncoder(splenc_1, splenc_2, 2, 2))

    vmha = VIDONMHA(64, 32, 4, 256, 4, 256, 4)
    out = vmha(encoder(x0))
    assert out.shape == (x0.shape[0], 32 * 4)

    return


def test_VIDON():
    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
        x_transform=ToDType("default"),
        y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    x0, _ = next(iter(dataloader))

    encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(torch.LongTensor(list(range(6))), -2, 2))

    mlp_1, mlp_2 = [MLP([2, 64], activation=nn.ReLU()) for _ in range(2)]
    split_additive = SplitAdditive(mlp_1, mlp_2, 2, 2)

    mha = VIDONMHA(64, 32, 4, 256, 4, 245, 4)

    processor = MLP([4*32, 4*32, 32], activation=nn.ReLU())

    vidon = VIDON(split_additive, mha, processor)

    assert vidon(encoder(x0)).shape == (x0.shape[0], 32)

    return

def test_VIDON_deeponet():
    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dset = FEniCSDataset(x_data, y_data, 
        x_transform=ToDType("default"),
        y_transform=ToDType("default"))
    dataloader = DataLoader(dset, batch_size=2, shuffle=False)

    uh, _ = next(iter(dataloader))

    branch_encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(torch.LongTensor(list(range(6))), -2, 2))

    mlp_1, mlp_2 = [MLP([2, 64], activation=nn.ReLU()) for _ in range(2)]
    split_additive = SplitAdditive(mlp_1, mlp_2, 2, 2)

    mha = VIDONMHA(64, 32, 4, 256, 4, 245, 4)

    processor = MLP([4*32, 4*32, 32], activation=nn.ReLU())

    branch_vidon = VIDON(split_additive, mha, processor)

    from neuraloperators.encoders import IdentityEncoder
    trunk_encoder = IdentityEncoder()

    trunk_net = MLP([2, 256, 256, 32], activation=nn.ReLU())

    from neuraloperators.deeponet import DeepONet


    deeponet = DeepONet(branch_encoder, branch_vidon, trunk_encoder, trunk_net,
                        U_dim=2, V_dim=2, final_bias=None, combine_style=2)

    eval_points = y_data.dof_coordinates[None,...].to(torch.get_default_dtype())
    out = deeponet(uh, eval_points)

    assert deeponet(uh, eval_points).shape == (uh.shape[0], eval_points.shape[1], deeponet.V_dim)

    return


if __name__ == "__main__":
    test_split_additive()
    test_VIDONMHAHead()
    test_VIDONMHA()
    test_VIDON()
    test_VIDON_deeponet()
