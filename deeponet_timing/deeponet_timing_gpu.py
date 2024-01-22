import argparse
import torch
import torch.nn as nn
import dolfin as df
import numpy as np

from timeit import default_timer as timer

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from dataset.dataset import MeshData
from tools.translation import torch_to_dolfin
from tqdm import tqdm

def create_mask_function(V, f: str | float = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1",
                             normalize: bool = True) -> torch.Tensor:

        """
            -Delta u = f in Omega
                u = 0 on dOmega
        """


        def boundary(x, on_boundary):
            return on_boundary
        u0 = df.Constant(0.0)
        
        bc = df.DirichletBC(V, u0, boundary)

        f_func = df.Expression(f, degree=5) if isinstance(f, str) else df.Constant(f)

        u = df.TrialFunction(V)
        v = df.TestFunction(V)

        a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
        l = f_func * v * df.dx

        uh = df.Function(V)
        df.solve(a == l, uh, bc)
        
        if normalize:
            uh.vector()[:] /= np.max(uh.vector()[:]) # Normalize mask to have sup-norm 1.

        uh = torch.tensor(uh.vector().get_local(), dtype=torch.get_default_dtype())[:,None]

        return uh


def time_deeponet(problem_dict: dict) -> None:

    deeponet, _, _, dset, \
    _, _, _, _, mask_tensor = load_deeponet_problem(problem_dict)

    x_data: MeshData = dset.x_data
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
    net.to(torch.device("cuda"))

    # deeponet.trunk.load_state_dict(torch.load(state_dict_trunk, map_location=torch.device("cpu")))
    # deeponet.branch.load_state_dict(torch.load(state_dict_branch, map_location=torch.device("cpu")))

    in_file = df.XDMFFile("dataset/learnext_period_p2/input.xdmf")

    msh = df.Mesh()
    in_file.read(msh)
    V = df.VectorFunctionSpace(msh, "CG", 2)
    V_scal = df.FunctionSpace(msh, "CG", 2)
    u_in = df.Function(V)
    in_file.read_checkpoint(u_in, "uh", 0)

    evaluation_points = torch.tensor(V_scal.tabulate_dof_coordinates(), dtype=torch.float32, device=torch.device("cuda"))

    mask_np = create_mask_function(V_scal).detach().numpy()

    u_out = np.zeros((V_scal.dim(), 2))
    def CG_vector_to_array(u: df.Function, u_out: np.ndarray | None = None) -> np.ndarray:
        """
        Layout: Columns ``(u_x, u_y)``
        """
        if u_out is None:
            u_out = np.zeros((u.function_space().dim() // 2, 2))

        raw_array_base = u.vector().get_local()
        u_out[:,0] = raw_array_base[0::2]
        u_out[:,1] = raw_array_base[1::2]

        return u_out
    
    N = 100
    start = timer()
    for _ in range(N):
        CG_vector_to_array(u_in, u_out)
        uh_torch = torch.tensor(u_out, dtype=torch.float32, device=torch.device("cuda"))
    end = timer()
    A = (end-start)/N
    print(f"A: {A:.2e}")


    with torch.no_grad():
        N = 20
        start = timer()
        for _ in range(N):
            corr_t = deeponet(uh_torch, evaluation_points)
        end = timer()
        B = (end-start)/N
        print(f"B: {B:.2e}")

        # N = 100
        # start = timer()
        # for _ in range(N):
        #     enc = deeponet.branch_encoder(uh_torch)
        # end = timer()
        # print(f"X: {(end-start)/N:.2e}")

        N = 100
        start = timer()
        for _ in range(N):
            corr_np = corr_t.detach().cpu().double().numpy()
            corr_np = corr_np * mask_np
        end = timer()
        C = (end-start)/N
        print(f"C: {C:.2e}")

    N = 100
    start = timer()
    for _ in range(N):
        new_dofs = u_in.vector().get_local()
        new_dofs[0::2] += corr_np[:,0]
        new_dofs[1::2] += corr_np[:,1]
        u_in.vector().set_local(new_dofs)
    end = timer()
    D = (end-start)/N
    print(f"D: {D:.2e}")


    return A, B, C, D


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-problem-file", default=Path("problems/default.json"), type=Path)
    parser.add_argument("--save-to-file", default=Path("output/data/deeponet_timing_gpu.npy"), type=Path)
    args = parser.parse_args()

    base_problem_file: Path = args.base_problem_file
    save_to_file: Path = args.save_to_file

    import json
    with open(base_problem_file, "r") as infile:
        problem_dict = json.loads(infile.read())

    dd = [3, 4, 5, 6, 7]
    ww = [128, 256, 512, 1024]
    pp = [16]

    # dd, ww = dd[:-2], ww[:-2]
    # dd = [2, 3]
    # ww = [1024, 2048, 4196]

    timing_arr = np.zeros((len(dd), len(ww), len(pp), 4))
    for i, d in enumerate(dd):
        for j, w in enumerate(ww):
            for k, p in enumerate(pp):
                problem_dict["branch"]["MLP"]["widths"][1:-1] = [412] + d * [w] + [p]
                problem_dict["trunk"]["MLP"]["widths"][1:-1] = [2] + d * [w] + [p]

                print(f"{d = :3d}, {w = :3d}, {p = :3d}")
                timing_arr[i,j,k] = time_deeponet(problem_dict)
                print()

    np.save(save_to_file, timing_arr)

    return


if __name__ == "__main__":
    main()
