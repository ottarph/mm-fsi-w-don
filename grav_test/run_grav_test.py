import dolfin as df
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from pathlib import Path

datapath = Path("grav_test/data/max_deformations_redo.xdmf")

datafile = df.XDMFFile(str(datapath))

msh = df.Mesh()
datafile.read(msh)


V_input = df.VectorFunctionSpace(msh, "CG", 2)
u_input = df.Function(V_input)

p = 2
V = df.VectorFunctionSpace(msh, "CG", p)
V_scal = df.FunctionSpace(msh, "CG", p)


dof_coords_file_path = Path(f"grav_test/data/learnext_inner_dof_coords.cg1.txt")
dof_coords = np.loadtxt(dof_coords_file_path)

results_dir = Path("hyperparameter_study/best_run")
problem_file = results_dir / "problem.json"
branch_state_dict =  results_dir / "branch.pt"
trunk_state_dict =  results_dir / "trunk.pt"


# Load DeepONet

from neuraloperators.loading import load_deeponet_problem

deeponet, _, _, _, \
    _, _, _, _, _ = load_deeponet_problem(problem_file)

with open(problem_file, "r") as infile:
    problem_dict = json.load(infile)


deeponet.branch.load_state_dict(torch.load(branch_state_dict, map_location="cpu"))
deeponet.trunk.load_state_dict(torch.load(trunk_state_dict, map_location="cpu"))


# Switch to correct branch encoder, due to different ordering along inner boundary between meshes

from neuraloperators.encoders import SequentialEncoder, ExtractBoundaryDofEncoder, FlattenEncoder
new_branch_encoder = SequentialEncoder(ExtractBoundaryDofEncoder(V, dof_coords_file_path), FlattenEncoder(start_dim=-2))
deeponet.branch_encoder = new_branch_encoder


filter_tensor_np = new_branch_encoder.encoders[0].filter_tensor.detach().numpy()
print(f"{np.linalg.norm(V_scal.tabulate_dof_coordinates()[filter_tensor_np] - dof_coords) = :.2e}")

print(f"{filter_tensor_np.shape = }")


# Compute mask function

f_str = problem_dict["mask_function_f"]
f_expr = df.Expression(f_str, degree=5)

trial = df.TrialFunction(V_scal)
test = df.TestFunction(V_scal)
a = df.inner(df.nabla_grad(trial), df.nabla_grad(test)) * df.dx
l = f_expr * test * df.dx
bc = df.DirichletBC(V_scal, df.Constant(0.0), "on_boundary")

u_mask = df.Function(V_scal)
df.solve(a == l, u_mask, bc)
mask_np = u_mask.vector().get_local()[:,None]
mask_np /= np.max(mask_np) # Normalize mask to have sup-norm 1.


eval_points_np = V_scal.tabulate_dof_coordinates()
eval_points_torch = torch.tensor(eval_points_np, dtype=torch.float32)




biharm_ext_file_path = Path("grav_test/data/biharm_ext.xdmf")
biharm_ext_file_path.unlink(missing_ok=True)
biharm_ext_file_path.with_suffix(".h5").unlink(missing_ok=True)
biharm_ext_file = df.XDMFFile(str(biharm_ext_file_path))
biharm_ext_file.write(msh)

harm_ext_file_path = Path("grav_test/data/harm_ext.xdmf")
harm_ext_file_path.unlink(missing_ok=True)
harm_ext_file_path.with_suffix(".h5").unlink(missing_ok=True)
harm_ext_file = df.XDMFFile(str(harm_ext_file_path))
harm_ext_file.write(msh)

don_ext_file_path = Path("grav_test/data/don_ext.xdmf")
don_ext_file_path.unlink(missing_ok=True)
don_ext_file_path.with_suffix(".h5").unlink(missing_ok=True)
don_ext_file = df.XDMFFile(str(don_ext_file_path))
don_ext_file.write(msh)



biharm_signs_file_path = Path("grav_test/data/biharm_signs.xdmf")
biharm_signs_file_path.unlink(missing_ok=True)
biharm_signs_file_path.with_suffix(".h5").unlink(missing_ok=True)
biharm_signs_file = df.XDMFFile(str(biharm_signs_file_path))
biharm_signs_file.write(msh)

harm_signs_file_path = Path("grav_test/data/harm_signs.xdmf")
harm_signs_file_path.unlink(missing_ok=True)
harm_signs_file_path.with_suffix(".h5").unlink(missing_ok=True)
harm_signs_file = df.XDMFFile(str(harm_signs_file_path))
harm_signs_file.write(msh)

don_signs_file_path = Path("grav_test/data/don_signs.xdmf")
don_signs_file_path.unlink(missing_ok=True)
don_signs_file_path.with_suffix(".h5").unlink(missing_ok=True)
don_signs_file = df.XDMFFile(str(don_signs_file_path))
don_signs_file.write(msh)



from grav_test.extensions import BiharmonicExtension, HarmonicExtension, DeepONetExtension

biharmonic_extension = BiharmonicExtension(V)
harmonic_extension = HarmonicExtension(V)
deeponet_extension = DeepONetExtension(V, deeponet, eval_points_np, mask_np)

from grav_test.degeneracy_check import get_degenerate_cells
from tools.mesh_quality import MeshQuality
scaled_jacobian = MeshQuality(msh, "scaled_jacobian")

harm_signed_mq_arr = np.zeros((3, msh.num_cells()))
biharm_signed_mq_arr = np.zeros((3, msh.num_cells()))
don_signed_mq_arr = np.zeros((3, msh.num_cells()))

u_input_V = df.Function(V)
for k in range(3):
    datafile.read_checkpoint(u_input, "uh", k)
    u_input_V.interpolate(u_input)
    bc = df.DirichletBC(V, u_input_V, "on_boundary")


    u_h = harmonic_extension.extend(u_input_V)
    u_b = biharmonic_extension.extend(u_input_V)
    u_don = deeponet_extension.extend(u_input_V)

    signs_h = get_degenerate_cells(u_h, tensor_dg_order=6)
    signs_b = get_degenerate_cells(u_b, tensor_dg_order=6)
    signs_don = get_degenerate_cells(u_don, tensor_dg_order=6)

    harm_ext_file.write_checkpoint(u_h, "uh", k, append=True)
    biharm_ext_file.write_checkpoint(u_b, "uh", k, append=True)
    don_ext_file.write_checkpoint(u_don, "uh", k, append=True)

    harm_signs_file.write_checkpoint(signs_h, "Sign", k, append=True)
    biharm_signs_file.write_checkpoint(signs_b, "Sign", k, append=True)
    don_signs_file.write_checkpoint(signs_don, "Sign", k, append=True)

    harm_mq = scaled_jacobian(u_h)
    biharm_mq = scaled_jacobian(u_b)
    don_mq = scaled_jacobian(u_don)

    signed_harm_mq = harm_mq * signs_h.vector()[:]
    signed_biharm_mq = biharm_mq * signs_b.vector()[:]
    signed_don_mq = don_mq * signs_don.vector()[:]

    harm_signed_mq_arr[k,:] = signed_harm_mq
    biharm_signed_mq_arr[k,:] = signed_biharm_mq
    don_signed_mq_arr[k,:] = signed_don_mq

np.save("grav_test/data/harm_signed_mq_arr.npy", harm_signed_mq_arr)
np.save("grav_test/data/biharm_signed_mq_arr.npy", biharm_signed_mq_arr)
np.save("grav_test/data/don_signed_mq_arr.npy", don_signed_mq_arr)

from grav_test.make_histograms import make_histograms_shorter_allpos

fig, axs = make_histograms_shorter_allpos(biharm_signed_mq_arr, don_signed_mq_arr)
fig.savefig("grav_test/figures/grav_test_histograms_shorter_allpos.pdf")

