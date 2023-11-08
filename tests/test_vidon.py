from neuraloperators.vidon import *

from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

from neuraloperators.networks import MLP
from neuraloperators.encoders import SequentialEncoder, CoordinateInsertEncoder, BoundaryFilterEncoder, FixedFilterEncoder, SplitAdditiveEncoder



def test_VIDONMHAHead():
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
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
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
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


def test_vidon():

    assert False

    return



if __name__ == "__main__":
    test_VIDONMHAHead()
    test_VIDONMHA()
    test_VIDON()