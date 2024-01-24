import dolfin as df
import numpy as np

from dataset.dataset import load_MeshData

from pathlib import Path

mesh_path = Path("dataset/learnext_period_p1/output.xdmf")
save_path = Path("grav_test/data/learnext_inner_dof_coords")

xdmf_file = df.XDMFFile(str(mesh_path))

msh = df.Mesh()
xdmf_file.read(msh)

def inner_boundary(x, on_boundary):
    if on_boundary:
        eps = 1e-3
        if df.near(x[1], 0, eps) or df.near(x[1], 0.41, eps) \
            or df.near(x[0], 0, eps) or df.near(x[0], 2.5, eps):
            return False
        else:
            return True
    else:
        return False

V_CG1 = df.FunctionSpace(msh, "CG", 1)
u_cg1 = df.Function(V_CG1)

bc = df.DirichletBC(V_CG1, df.Constant(1), inner_boundary)
bc.apply(u_cg1.vector())

ids = np.flatnonzero(u_cg1.vector().get_local())
print(ids)

print(V_CG1.tabulate_dof_coordinates()[ids])
print(V_CG1.tabulate_dof_coordinates()[ids].shape)

np.savetxt(save_path.with_suffix(".cg1.txt"), V_CG1.tabulate_dof_coordinates()[ids])


V_CG2 = df.FunctionSpace(msh, "CG", 2)
u_cg2 = df.Function(V_CG2)

bc = df.DirichletBC(V_CG2, df.Constant(1), inner_boundary)
bc.apply(u_cg2.vector())

ids = np.flatnonzero(u_cg2.vector().get_local())
print(ids)

print(V_CG2.tabulate_dof_coordinates()[ids])
print(V_CG2.tabulate_dof_coordinates()[ids].shape)

np.savetxt(save_path.with_suffix(".cg2.txt"), V_CG2.tabulate_dof_coordinates()[ids])
