import torch
import torch.nn as nn


from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
from torch.utils.data import DataLoader

x_data, y_data = load_MeshData("dataset/artificial_learnext", "XDMF")
dset = FEniCSDataset(x_data, y_data, 
                x_transform=ToDType("default"),
                y_transform=ToDType("default"))
dataloader = DataLoader(dset, batch_size=2, shuffle=False)

x0, _ = next(iter(dataloader))


from neuraloperators.encoders import CoordinateInsertEncoder, BoundaryFilterEncoder, \
      RandomPermuteEncoder, SequentialEncoder, SplitAdditiveEncoder, FixedFilterEncoder
from neuraloperators.networks import MLP

splenc_1 = MLP([2, 64], nn.ReLU())
splenc_2 = MLP([2, 64], nn.ReLU())
encoder = SequentialEncoder(CoordinateInsertEncoder(x_data), BoundaryFilterEncoder(x_data), FixedFilterEncoder(torch.LongTensor(list(range(6))), -2, 2), SplitAdditiveEncoder(splenc_1, splenc_2, 2, 2))



# foo = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64, dropout=0.0, batch_first=True)
foo = nn.MultiheadAttention(embed_dim = 64, num_heads = 8, dropout = 0.0, batch_first = True)

print(f"{x0.shape = }")
print(f"{encoder(x0).shape = }")
# print(f"{foo(encoder(x0)).shape = }")

q, v, k = torch.rand(encoder(x0).shape), torch.rand(encoder(x0).shape), torch.rand(encoder(x0).shape)
print(foo(q, v, k))
print(f"{foo(q, v, k)[0].shape = }")
print(f"{foo(q, v, k)[1].shape = }")

print(torch.sum(foo(q, v, k)[1], dim=-1))

print(foo(q, v, k, need_weights=False)[0].shape)


class VIDONMHAHead(nn.Module):
    
    def __init__(self, d_enc: int, )

class VIDONMHA(nn.Module):
    
    def __init__(self, d_enc: int, num_heads: int):
        super().__init__()



        return
    

