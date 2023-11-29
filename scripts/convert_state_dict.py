import argparse
import torch
import torch.nn as nn
import numpy as np
import shutil

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader
from dataset.dataset import MeshData
from neuraloperators.training import save_model, load_model

def convert_state_dict(problem_file: Path, results_dir: Path) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deeponet, train_dataloader, val_dataloader, dset, \
    optimizer, scheduler, loss_fn, num_epochs, mask_tensor = load_deeponet_problem(problem_file)

    x_data: MeshData = dset.x_data
    y_data: MeshData = dset.y_data

    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.cost_functions import DataInformedLoss
    x0, y0 = next(iter(train_dataloader))
    z0 = x0 + deeponet(x0, evaluation_points) * mask_tensor
    if isinstance(loss_fn, DataInformedLoss):
        loss_fn(x0, y0, z0)
    else:
        loss_fn(z0, y0)


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
    loss_fn.to(device)

    context = Context(net, loss_fn, optimizer, scheduler)
    context.load_model(results_dir)

    print(context.final_val_loss)

    deeponet: DeepONet = context.network.deeponet
    print(deeponet)

    save_model(deeponet.branch, results_dir / "branch.pt")
    save_model(deeponet.trunk, results_dir / "trunk.pt")

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("problems/default.json"), type=Path)
    parser.add_argument("--results-dir", default=Path("results/default"), type=Path)
    # parser.add_argument("--problem-file", default=Path("problems/gamma.json"), type=Path)
    # parser.add_argument("--results-dir", default=Path("results/gamma"), type=Path)
    # parser.add_argument("--problem-file", default=Path("problems/defaultdeepsets.json"), type=Path)
    # parser.add_argument("--results-dir", default=Path("results/defaultdeepsets"), type=Path)
    # parser.add_argument("--problem-file", default=Path("problems/defaultvidon.json"), type=Path)
    # parser.add_argument("--results-dir", default=Path("results/defaultvidon"), type=Path)
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    results_dir: Path = args.results_dir


    if not problem_file.is_file():
        raise RuntimeError("Given problem description file is not a file.")
    
    convert_state_dict(problem_file, results_dir)

    return


if __name__ == "__main__":
    main()
