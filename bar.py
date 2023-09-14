import torch
import numpy as np

from neuraloperators.loading import load_deeponet_problem


problem_path = "problems/problem.json"

deeponet, dataloader = load_deeponet_problem(problem_path)

print(deeponet)
print(dataloader)

x0, y0 = next(iter(dataloader))
x0, y0 = x0.float(), y0.float()

print(deeponet)

eval_points = dataloader.dataset.y_data.dof_coordinates[None,...].float()
print(eval_points)

z0 = deeponet(x0, eval_points)

print(z0.shape)



xx = dataloader.dataset.x_data.boundary_dof_coordinates.numpy()
uu = x0[0,...].double().detach().numpy()
print(xx.shape)
print(uu.shape)

cc = np.linalg.norm(uu, axis=1)
print(cc.shape)

print(uu)

import matplotlib.pyplot as plt
plt.figure()

plt.scatter(xx[:,0] + uu[:,0], xx[:,1] + uu[:,1], c=cc, cmap="viridis")

plt.savefig("foo.pdf")

