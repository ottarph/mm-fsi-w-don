import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer
from tqdm import tqdm

import itertools

mesh_mult = 1
num_rep = min(1000, int(1000 / mesh_mult))
dset_b = torch.rand((1000, int(3935*mesh_mult), 2))
dset_t = torch.rand((1000, int(3935*mesh_mult), 2))

print(f"{mesh_mult = }")
print(f"{num_rep = }")

start = timer()
for k in range(len(dset_b)):
    b, t = dset_b[k], dset_t[k]
    b_c, t_c = b.to("cuda"), t.to("cuda")
    bc = (b_c + t_c).to("cpu")
end = timer()
print(f"to CUDA, {(end-start)/len(dset_b):.2e}")
bc.sum()


dset = TensorDataset(dset_b, dset_t)
dl = DataLoader(dset, batch_size=1, shuffle=False)


from neuraloperators.deeponet import DeepONet
from neuraloperators.networks import MLP
from neuraloperators.encoders import FixedFilterEncoder, FlattenEncoder, SequentialEncoder, IdentityEncoder

filter_tens = torch.randperm(206)
branch_encoder = SequentialEncoder(FixedFilterEncoder(filter_tens, dim=-2, unit_shape_length=2), FlattenEncoder(start_dim=-2))
trunk_encoder = IdentityEncoder()

d = 7
w = 512
p = 16
bwidth = [412] + [w]*d + [p]
branch = MLP(bwidth)
twidth = [2] + [w]*d + [p]
trunk = MLP(twidth)

deeponet = DeepONet(branch_encoder, branch, trunk_encoder, trunk, 2, 2)

dliter = iter(dl)
b, t = next(dliter)

print(f"{b.shape = }")
print(f"{t.shape = }")

print("------+------+------+------+------+------+------+------")

Tot = 0.0

N = num_rep
start = timer()
for _ in range(N):
    benc = branch_encoder(b)
end = timer()
print(f"CPU: branch_encoder(b): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    tenc = trunk_encoder(t)
end = timer()
print(f"CPU: trunk_encoder(t): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    bk = branch(benc)
end = timer()
print(f"CPU: branch(benc): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    tk = trunk(tenc)
end = timer()
print(f"CPU: trunk(tenc): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    dn = deeponet.combine(bk, tk)
end = timer()
print(f"CPU: deeponet.combine(bk, tk): {(end-start)/N:.2e}")
Tot += (end-start)/N

print(f"CPU: Tot: {Tot:.2e}")

N = num_rep
start = timer()
for _ in range(N):
    dn = deeponet(b, t)
end = timer()
print(f"CPU: deeponet(b, t): {(end-start)/N:.2e}")

print("------+------+------+------+------+------+------+------")

torch.cuda.empty_cache()
deeponet.to("cuda")
filter_tens = filter_tens.to("cuda")
deeponet.branch_encoder = SequentialEncoder(FixedFilterEncoder(filter_tens, dim=-2, unit_shape_length=2), FlattenEncoder(start_dim=-2))

dliter = iter(dl)
b, t = next(dliter)
# b, t = b.to("cuda"), t.to("cuda")

Tot = 0.0

torch.cuda.empty_cache()

N = num_rep
start = timer()
for _ in range(N):
    b_c= b.to("cuda")
end = timer()
print(f"GPU: b.to(\"cuda\"): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    t_c = t.to("cuda")
end = timer()
print(f"GPU: t.to(\"cuda\"): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    benc = branch_encoder(b_c)
end = timer()
print(f"GPU: branch_encoder(b): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    tenc = trunk_encoder(t_c)
end = timer()
print(f"GPU: trunk_encoder(t): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    bk = branch(benc)
end = timer()
print(f"GPU: branch(benc): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    tk = trunk(tenc)
end = timer()
print(f"GPU: trunk(tenc): {(end-start)/N:.2e}")
Tot += (end-start)/N

N = num_rep
start = timer()
for _ in range(N):
    dn = deeponet.combine(bk, tk)
    # dn = deeponet.combine_2(bk, tk)
end = timer()
print(f"GPU: deeponet.combine(bk, tk): {(end-start)/N:.2e}")
# print(f"GPU: deeponet.combine_2(bk, tk): {(end-start)/N:.2e}")
Tot += (end-start)/N

print(f"GPU: Tot: {Tot:.2e}")

torch.cuda.empty_cache()
N = num_rep
start = timer()
for _ in range(N):
    b_c, t_c = b.to("cuda"), t.to("cuda")
    dn = deeponet(b_c, t_c)
end = timer()
print(f"GPU: deeponet(b, t)+to(\"cuda\"): {(end-start)/N:.2e}")

print("\n")
