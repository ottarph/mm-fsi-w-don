import torch
import torch.nn as nn

foo = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64, dropout=0.0, batch_first=True)

from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
dset = FEniCSDataset(x_data, y_data, 
                x_transform=ToDType("default"),
                y_transform=ToDType("default"))
dataloader = DataLoader(dset, batch_size=2, shuffle=False)

x0, _ = next(iter(dataloader))

print(f"{x0.shape = }")

from neuraloperators.encoders import CoordinateInsertEncoder, BoundaryFilterEncoder, \
      RandomPermuteEncoder, SequentialEncoder, SplitAdditiveEncoder
from neuraloperators.mlp import MLP

splenc_1 = MLP([2, 64], nn.ReLU())
splenc_2 = MLP([2, 64], nn.ReLU())
encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), RandomPermuteEncoder(-2, 2), SplitAdditiveEncoder(splenc_1, splenc_2, 2, 2))



print(foo(encoder(x0)))

