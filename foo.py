import json

with open("problems/problem.json", "r") as infile:
    problemdict = json.loads(infile.read())

print(problemdict)
print(problemdict["branch"])
print(problemdict["trunk"])
print(problemdict["dataset"])

from dataset.dataset import load_MeshData, FEniCSDataset, OnBoundary
from torch.utils.data import DataLoader
x_data, y_data = load_MeshData(problemdict["dataset"]["directory"], problemdict["dataset"]["style"])
dataset = FEniCSDataset(x_data, y_data, x_transform=OnBoundary(x_data))
dataloader = DataLoader(dataset, batch_size=problemdict["dataset"]["batch_size"])
x0, y0 = next(iter(dataloader))
print(x0.shape)
print(y0.shape)

sensors = x_data.boundary_dof_coordinates
print(sensors.shape)

from neuraloperators.loading import build_model

branch_net = build_model(problemdict["branch"])
trunk_net = build_model(problemdict["trunk"])

from neuraloperators.deeponet import BranchNetwork, TrunkNetwork, DeepONet
branch = BranchNetwork(branch_net, sensors, 2, 2, 2, 2, problemdict["width"])
trunk = TrunkNetwork(trunk_net, 2, 2, 2, 2, problemdict["width"])

print(branch)
print(trunk)

from neuraloperators.deeponet import DeepONet
deeponet = DeepONet(branch, trunk, sensors, problemdict["final_bias"])

print(deeponet)
