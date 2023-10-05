import argparse
import torch
import torch.nn as nn
import numpy as np
import shutil

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader
from dataset.dataset import MeshData, OnBoundary

def run_boundary_problem(problem_file: Path, results_dir: Path, xdmf_overwrite: bool = False) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deeponet, train_dataloader, val_dataloader, dataset, \
    optimizer, scheduler, loss_fn, num_epochs, mask_tensor = load_deeponet_problem(problem_file)

    x_data: MeshData =  dataset.x_data
    y_data: MeshData =  dataset.y_data

    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())
    boundary_filter = OnBoundary(x_data)

    from neuraloperators.cost_functions import DataInformedLoss
    x0, y0 = next(iter(train_dataloader))
    z0 = x0 + deeponet(boundary_filter(x0), evaluation_points) * mask_tensor
    if isinstance(loss_fn, DataInformedLoss):
        loss0 = loss_fn(x0, y0, z0)
    else:
        loss0 = loss_fn(z0, y0)


    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(nn.Module):
        def __init__(self, deeponet: DeepONet, evaluation_points: torch.Tensor,
                     boundary_filter: OnBoundary, mask_tensor: torch.Tensor):
            super().__init__()

            self.deeponet = deeponet
            self.boundary_filter = boundary_filter

            if len(mask_tensor.shape) == 1:
                mask_tensor = mask_tensor[:,None]

            self.register_buffer("evaluation_points", evaluation_points)
            self.register_buffer("mask_tensor", mask_tensor)
            self.evaluation_points: torch.Tensor
            self.mask_tensor: torch.Tensor
            self.evaluation_points.requires_grad_(False)
            self.mask_tensor.requires_grad_(False)

            return
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.deeponet(self.boundary_filter(x), self.evaluation_points) * self.mask_tensor

    net = EvalWrapper(deeponet, evaluation_points, boundary_filter, mask_tensor)
    net.to(device)

    context = Context(net, loss_fn, optimizer, scheduler)

    train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader=val_dataloader)

    context.save_results(results_dir)
    context.save_summary(results_dir)
    context.plot_results(results_dir)
    context.save_model(results_dir)
    shutil.copy(problem_file, results_dir / "problem.json")

    latest_results_dir = Path("results/latest")
    context.save_results(latest_results_dir)
    context.save_summary(latest_results_dir)
    context.plot_results(latest_results_dir)
    context.save_model(latest_results_dir)
    shutil.copy(problem_file, latest_results_dir / "problem.json")


    net.to("cpu")
    x0, y0 = next(iter(train_dataloader))
    x0, y0 = x0[[0],...], y0[[0],...]
    coords = x_data.dof_coordinates
    z0 = net(x0)


    from tools.xdmf_io import pred_to_xdmf

    pred_to_xdmf(net, dataset, results_dir / "pred", overwrite=xdmf_overwrite)
    shutil.copy(results_dir / "pred.xdmf", latest_results_dir / "pred.xdmf")
    shutil.copy(results_dir / "pred.h5", latest_results_dir / "pred.h5")


    import matplotlib.pyplot as plt

    input_x = coords.detach().numpy()
    input_y = x0.numpy()[0,...]
    input_z = input_x + input_y
    plt.figure()
    plt.scatter(input_z[:,0], input_z[:,1], c=np.linalg.norm(input_y, axis=1))
    plt.savefig(results_dir / "viz_input.pdf")
    plt.savefig(latest_results_dir / "viz_input.pdf")

    output_x = coords.detach().numpy()
    output_y = y0.detach().numpy()[0,...]
    output_z = output_x + output_y
    plt.figure()
    plt.scatter(output_z[:,0], output_z[:,1], c=np.linalg.norm(output_y, axis=1))
    plt.savefig(results_dir / "viz_output.pdf")
    plt.savefig(latest_results_dir / "viz_output.pdf")

    pred_x = coords.detach().numpy()
    pred_y = z0.detach().numpy()[0,...]
    pred_z = pred_x + pred_y
    plt.figure()
    plt.scatter(pred_z[:,0], pred_z[:,1], c=np.linalg.norm(pred_y, axis=1))
    plt.savefig(results_dir / "viz_pred.pdf")
    plt.savefig(latest_results_dir / "viz_pred.pdf")

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("problems/default.json"), type=Path)
    parser.add_argument("--results-dir", default=Path("results/default"), type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    results_dir: Path = args.results_dir
    overwrite: bool = args.overwrite

    if not problem_file.is_file():
        raise RuntimeError("Given problem description file is not a file.")
    if results_dir.exists():
        if len(list(results_dir.iterdir())) > 0 and overwrite == False:
            print("Results directory is not empty. Continuing might overwrite data.")
            if not input("Continue? (y/n): ").lower() == "y":
                print("\nExiting program.")
                quit()
    else:
        results_dir.mkdir(parents=True)

    run_boundary_problem(problem_file, results_dir, xdmf_overwrite=True)

    return


if __name__ == "__main__":
    main()
