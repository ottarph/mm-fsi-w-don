import torch
import torch.nn as nn
import numpy as np

import json
with open("problems/problem.json", "r") as infile:
    problemdict = json.loads(infile.read())

print(problemdict)
print(problemdict["branch"])
print(problemdict["trunk"])
print(problemdict["dataset"])

from dataset.dataset import load_MeshData, FEniCSDataset, OnBoundary, ToDType
from torch.utils.data import DataLoader
x_data, y_data = load_MeshData(problemdict["dataset"]["directory"], problemdict["dataset"]["style"])
dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=nn.Sequential(OnBoundary(x_data), ToDType("default")),
                    y_transform=ToDType("default"))
dataloader = DataLoader(dataset, batch_size=1)
x0, y0 = next(iter(dataloader))
print(f"{x0.shape = }")
print(f"{y0.shape = }")
print(f"{x0.dtype = }")
print(f"{y0.dtype = }")

sensors = x_data.boundary_dof_coordinates
print(f"{sensors.shape = }")
print(f"{sensors.dtype = }")
evaluation_points = dataset.y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())
print(f"{evaluation_points.shape = }")
print(f"{evaluation_points.dtype = }")

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

z0 = deeponet(x0, evaluation_points)
print(f"{z0.shape = }")
print(f"{y0.shape = }")
print(f"{z0.dtype = }")
print(f"{y0.dtype = }")


import matplotlib.pyplot as plt


input_x = dataset.x_data.boundary_dof_coordinates.detach().numpy()
input_y = x0.detach().numpy()[0,...]
input_z = input_x + input_y

plt.figure()
plt.scatter(input_z[:,0], input_z[:,1], c=np.linalg.norm(input_y, axis=1))

plt.savefig("foo_input.pdf")


output_x = dataset.x_data.dof_coordinates.detach().numpy()
output_y = y0.detach().numpy()[0,...]
output_z = output_x + output_y

plt.figure()
plt.scatter(output_z[:,0], output_z[:,1], c=np.linalg.norm(output_y, axis=1))

plt.savefig("foo_output.pdf")


pred_x = dataset.y_data.dof_coordinates.detach().numpy()
pred_y = z0.detach().numpy()[0,...]
pred_z = pred_x + pred_y

plt.figure()
plt.scatter(pred_z[:,0], pred_z[:,1], c=np.linalg.norm(pred_y, axis=1))

plt.savefig("foo_pred.pdf")

