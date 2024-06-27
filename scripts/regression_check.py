

def setup():

    import subprocess as sp
    from pathlib import Path

    # Download data
    DATA_DIR_PATH = Path("../mm-fsi-w-don-DATA")

    Path("TMP_DATA_DIR").mkdir(parents=True, exist_ok=True)
    sp.run(["cp", "-a", str(DATA_DIR_PATH) + "/.", "TMP_DATA_DIR"], check=True)


    # Populate directories

    Path("tmp_output").mkdir(parents=True, exist_ok=True)

    # Get dataset to tmp_output/dataset
    sp.run(["mv", "TMP_DATA_DIR/learnext_dataset/learnext_period_p1", "tmp_output/dataset"], check=True)

    # Write reduced problem file to tmp_output/problem.json
    import json
    with open("hyperparameter_study/base_problem.json", "r") as infile:
        problem = json.load(infile)
    problem["num_epochs"] = 3
    problem["branch"]["MLP"]["widths"] = [412] + [64] + [32]
    problem["trunk"]["MLP"]["widths"] = [2] + [64] + [32]
    with open("tmp_output/problem.json", "w") as outfile:
        json.dump(problem, outfile, indent=4)

    # Get mesh to tmp_output/mesh
    sp.run(["mv", "TMP_DATA_DIR/mesh", "tmp_output/mesh"], check=True)

    # Get warmstart_state to tmp_output/warmstart_state
    sp.run(["mv", "TMP_DATA_DIR/warmstart/state", "tmp_output/warmstart_state"], check=True)

    # Get pretrained model to tmp_output/model
    sp.run(["mv", "TMP_DATA_DIR/best_run_model", "tmp_output/model"], check=True)


    # Remove temporary directory
    sp.run(["rm", "-r", "TMP_DATA_DIR"], check=True)

    return





def train_network():

    import torch
    import torch.nn as nn
    import numpy as np
    import shutil

    from pathlib import Path

    from neuraloperators.loading import load_deeponet_problem
    from neuraloperators.training import Context, train_with_dataloader, save_model
    from dataset.dataset import MeshData

    device = "cpu"

    problem_file = Path("tmp_output/problem.json")
    results_dir = Path("tmp_output/training_results")

    deeponet, train_dataloader, val_dataloader, dset, \
    optimizer, scheduler, loss_fn, num_epochs, mask_tensor = load_deeponet_problem(problem_file)

    y_data: MeshData = dset.y_data

    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())


    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(nn.Module):
        def __init__(self, deeponet: DeepONet, evaluation_points: torch.Tensor, mask_tensor: torch.Tensor):
            super().__init__()

            self.deeponet = deeponet

            if len(mask_tensor.shape) == 1:
                mask_tensor = mask_tensor[:,None]

            self.register_buffer("evaluation_points", evaluation_points)
            self.register_buffer("mask_tensor", mask_tensor)
            self.evaluation_points: torch.Tensor
            self.mask_tensor: torch.Tensor
            self.evaluation_points.requires_grad_(False)
            self.mask_tensor.requires_grad_(False)

            return
        
        def forward(self, uh: torch.Tensor) -> torch.Tensor:
            return uh + self.deeponet(uh, self.evaluation_points) * self.mask_tensor

    
    net = EvalWrapper(deeponet, evaluation_points, mask_tensor)
    net.to(device)

    context = Context(net, loss_fn, optimizer, scheduler)
    
    show_mb_pbar = False


    train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader=val_dataloader,
                            show_minibatch_pbar=show_mb_pbar)

    results_dir.mkdir(parents=True, exist_ok=True)
    save_model(deeponet.branch, results_dir / "branch.pt")
    save_model(deeponet.trunk, results_dir / "trunk.pt")
    shutil.copy(problem_file, results_dir / "problem.json")

    

    return


def run_FSI():

    import dolfin as df
    import numpy as np
    import matplotlib.pyplot as plt
    import sys, os

    from pathlib import Path

    import FSIsolver.fsi_solver.solver as solver



    MESH_DIR = Path("tmp_output/mesh")

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
    FSI_param['T'] = 15.0 + 1 * FSI_param["deltat"] # Will then compute 2 timesteps
    FSI_param["save_int"] = 0.3 * FSI_param["deltat"]

    FSI_param['displacement_point'] = df.Point((0.6, 0.2))

    # boundary conditions, need to be 0 at t = 0
    Ubar = 1.0
    FSI_param['boundary_cond'] = df.Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                            "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                            Ubar=Ubar, t=FSI_param['t'], degree=2)


    from deeponet_extension.extension import DeepONetExtension
    DEEPONET_DIR = Path("tmp_output/model")
    extension_operator = DeepONetExtension(fluid_domain, DEEPONET_DIR, T_switch=0.0, torch_device="cpu")

    output_dir = Path("tmp_output/fsi")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save options
    FSI_param["save_data_dir"] = str(output_dir / "data")
    FSI_param["save_state_dir"] = str(output_dir / "data")
    FSI_param["save_snapshots_dir"] = str(output_dir / "data")
    FSI_param["save_data_on"] = True
    FSI_param["save_states_on"] = False
    FSI_param["save_snapshot_on"] = False
    FSI_param["warmstart_state_dir"] = str(Path("tmp_output/warmstart_state"))
    # FSI_param["warmstart_test_dir"] = str(output_dir / "data")
    WARMSTART = True

    fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=WARMSTART)

    fsisolver.solve()
    print("Solver complete")



    return


def check_results():

    from pathlib import Path

    output = Path("tmp_output")

    # Check output of NN-training exists
    nn_output  = output / "training_results"
    nn_output_check = nn_output.exists() and (nn_output / "branch.pt").exists() and (nn_output / "trunk.pt").exists() and (nn_output / "problem.json").exists()

    # Check output of FSI exists
    fsi_output = output / "fsi"
    fsi_data = fsi_output / "data"
    fsi_output_check = fsi_output.exists() and fsi_data.exists() and (fsi_data / "determinant.txt").exists() and (fsi_data / "displacementy.txt").exists() and \
        (fsi_data / "drag.txt").exists() and (fsi_data / "FSI_params.json").exists and (fsi_data / "lift.txt").exists() and (fsi_data / "times.txt").exists() and \
            (fsi_data / "warmstarted.txt").exists()


    
    return nn_output_check and fsi_output_check


def cleanup():
    import subprocess as sp

    sp.run(["rm", "-r", "tmp_output"], check=True)


    return


def main():

    success = False

    try:    
        setup()
        train_network()
        run_FSI()
        success = check_results()

    except:
        pass

    cleanup()

    print(f"Success: {success}")

    # if success:
    #     quit(0)
    # else:
    #     quit(1)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    quit(exit_code)
