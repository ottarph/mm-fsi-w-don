import argparse
import torch
import torch.nn as nn
import dolfin as df

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from dataset.dataset import MeshData
from tools.translation import torch_to_dolfin
from tqdm import tqdm

def save_trunk_basis(problem_file: Path, state_dict_path: Path, save_file: Path) -> None:

    deeponet, _, _, dset, \
    _, _, _, _, mask_tensor = load_deeponet_problem(problem_file)

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
    net.to(torch.device("cpu"))

    deeponet.trunk.load_state_dict(torch.load(state_dict_path, map_location=torch.device("cpu")))
    trunk_basis_tensor = deeponet.trunk(evaluation_points) * mask_tensor

    trunk_basis = trunk_basis_tensor.detach()[0,:,:]

    msh = y_data.mesh
    V = df.FunctionSpace(msh, "CG", y_data.function_space.ufl_element().degree())
    u_out = df.Function(V)

    out_file = df.XDMFFile(str(save_file))
    out_file.write(msh)

    for k in tqdm(range(trunk_basis.shape[1])):
        torch_to_dolfin(trunk_basis[:,k], V, u_out)
        out_file.write_checkpoint(u_out, "phi", k, append=True)

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("latest_results/problem.json"), type=Path)
    parser.add_argument("--state-dict", default=Path("latest_results/trunk.pt"), type=Path)
    parser.add_argument("--save-file", default=Path("output/fenics/latest.trunk_basis.xdmf"), type=Path)
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    state_dict: Path = args.state_dict
    save_file: Path = args.save_file

    possible_collisions = [save_file.with_suffix(".xdmf"), save_file.with_suffix(".h5")]
    collisions = list(filter(lambda p: p.exists(), possible_collisions))

    if len(collisions) > 0:
        print(f"Save files {[str(p) for p in collisions]} already exist.")
        if input("Overwrite? (y/n): ").lower() != "y":
            quit()
        else:
            [p.unlink() for p in collisions]


    save_trunk_basis(problem_file, state_dict, save_file)

    return


if __name__ == "__main__":
    main()
