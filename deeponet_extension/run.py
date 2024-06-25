import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from pathlib import Path

import FSIsolver.fsi_solver.solver as solver



MESH_DIR = Path("deeponet_extension/data/mesh")

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(MESH_DIR / "mesh_triangles.xdmf")) as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 1)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(MESH_DIR / "facet_mesh.xdmf")) as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(MESH_DIR / "mesh_triangles.xdmf")) as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)


# boundary parts
params = np.load(MESH_DIR / "params.npy", allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = df.MeshView.create(domains, params["fluid"])
solid_domain = df.MeshView.create(domains, params["solid"])


# parameters for FSI system
FSI_param = {}

FSI_param['fluid_mesh'] = fluid_domain
FSI_param['solid_mesh'] = solid_domain

FSI_param["material_model"] = "STVK"

FSI_param['lambdas'] = 2.0e6
FSI_param['mys'] = 0.5e6
FSI_param['rhos'] = 1.0e4
FSI_param['rhof'] = 1.0e3
FSI_param['nyf'] = 1.0e-3

FSI_param['t'] = 0.0
FSI_param['deltat'] = 0.0025
# FSI_param['deltat'] = 0.01
FSI_param['T'] = 15.1
FSI_param["save_int"] = 0.01

FSI_param['displacement_point'] = df.Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = df.Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)


from deeponet_extension.extension import DeepONetExtension
DEEPONET_DIR = Path("deeponet_extension/models/best_run")
extension_operator = DeepONetExtension(fluid_domain, DEEPONET_DIR, T_switch=0.0, torch_device="gpu")

# from FSIsolver.extension_operator.extension import Biharmonic
# extension_operator = Biharmonic(fluid_domain)

Path("output/fsi_run_don").mkdir(parents=True, exist_ok=True)

# save options
FSI_param["save_data_dir"] = str(Path("output/fsi_run_don/data"))
FSI_param["save_state_dir"] = str(Path("output/fsi_run_don/state"))
FSI_param["save_snapshots_dir"] = str(Path("output/fsi_run_don/snapshots"))
FSI_param["save_data_on"] = True
FSI_param["save_states_on"] = True
FSI_param["save_snapshot_on"] = True
FSI_param["warmstart_state_dir"] = str(Path("deeponet_extension_data/warmstart_state"))
FSI_param["warmstart_test_dir"] = str(Path("output/fsi_run_don/warmstart_test"))
WARMSTART = True
# df.set_log_active(False)

fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=WARMSTART) #warmstart needs to be set to False for the first run

fsisolver.solve()
print("Solver complete")
