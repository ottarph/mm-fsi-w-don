import torch
import torch.nn as nn


from neuraloperators.deeponet import DeepONet
from neuraloperators.networks import MLP
from neuraloperators.encoders import FixedFilterEncoder, FlattenEncoder, SequentialEncoder, IdentityEncoder

filter_tens = torch.randperm(206)
branch_encoder = SequentialEncoder(FixedFilterEncoder(filter_tens, dim=-2, unit_shape_length=2), FlattenEncoder(start_dim=-2))
trunk_encoder = IdentityEncoder()

d = 7
w = 256
p = 16
bwidth = [412] + [w]*d + [p]
branch = MLP(bwidth)
twidth = [2] + [w]*d + [p]
trunk = MLP(twidth)

deeponet = DeepONet(branch_encoder, branch, trunk_encoder, trunk, 2, 2)


model_param_counts = [sum(map(torch.numel, model.parameters())) for model in [deeponet, branch, trunk]]
print(model_param_counts)
