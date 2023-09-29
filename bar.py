import torch
import numpy as np

from neuraloperators.loading import load_deeponet_problem


problem_path = "problems/problem.json"

deeponet, dataloader = load_deeponet_problem(problem_path)

print(deeponet)
print(dataloader)

x0, y0 = next(iter(dataloader))
# x0, y0 = x0.float(), y0.float()

print(deeponet)

eval_points = dataloader.dataset.y_data.dof_coordinates[None,...].float()
print(eval_points)

# z0 = deeponet(x0, eval_points)

# print(z0.shape)



xx = dataloader.dataset.x_data.dof_coordinates.numpy()
uu = x0[0,...].double().detach().numpy()
print(np.count_nonzero(uu))
print(xx.shape)
print(uu.shape)

cc = np.linalg.norm(uu, axis=1)
print(cc.shape)



import matplotlib.pyplot as plt
plt.figure()

plt.scatter(xx[:,0] + uu[:,0], xx[:,1] + uu[:,1], c=cc, cmap="viridis")

plt.savefig("foo.pdf")

import dolfin as df
mesh = dataloader.dataset.x_data.mesh
V = df.VectorFunctionSpace(mesh, "CG", 2, 2)
u = df.Function(V)

new_dofs = np.zeros_like(u.vector().get_local())
new_dofs[0::2] = uu[:,0]
new_dofs[1::2] = uu[:,1]
u.vector().set_local(new_dofs)


file = df.File("foo.pvd")
file << u



