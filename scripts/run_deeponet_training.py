import argparse
import torch
import torch.nn as nn
import numpy as np
import shutil
import os

from pathlib import Path
from datetime import datetime
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader, load_model, save_model
from dataset.dataset import MeshData, FEniCSDataset, load_MeshData, ToDType

def run_boundary_problem(problem_file: Path, results_dir: Path, 
                         xdmf_overwrite: bool = False, save_xdmf: bool = True,
                         latest_results_dir: Path = Path("latest_results")) -> None:

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

    context = Context(net, loss_fn, optimizer, scheduler)
    
    show_mb_pbar = len(dset) > 20 * train_dataloader.batch_size

    try:
        train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader=val_dataloader,
                              show_minibatch_pbar=show_mb_pbar)
    except KeyboardInterrupt:
        if input("Training interrupted. Save current progress? (y/n): ").lower() != "y":
            quit()

    results_data_dir = results_dir / "data"
    context.save_results(results_data_dir)
    context.save_summary(results_data_dir)
    context.plot_results(results_dir)
    save_model(deeponet.branch, results_dir / "branch.pt")
    save_model(deeponet.trunk, results_dir / "trunk.pt")
    shutil.copy(problem_file, results_dir / "problem.json")

    latest_results_data_dir = latest_results_dir / "data"
    context.save_results(latest_results_data_dir)
    context.save_summary(latest_results_data_dir)
    context.plot_results(latest_results_dir)
    save_model(deeponet.branch, latest_results_dir / "branch.pt")
    save_model(deeponet.trunk, latest_results_dir / "trunk.pt")
    shutil.copy(problem_file, latest_results_dir / "problem.json")


    LOGDIR = Path("results/log")
    LOGENTRY = datetime.now().strftime(r"%Y_%m_%d-%H_%M")
    (LOGDIR / LOGENTRY).mkdir(exist_ok=True)
    shutil.copy(problem_file, LOGDIR / LOGENTRY / "problem.json")
    context.plot_results(LOGDIR / LOGENTRY)


    net.to("cpu")
    x0, y0 = next(iter(train_dataloader))
    x0, y0 = x0[[0],...], y0[[0],...]
    z0 = net(x0)


    test_dataset_path = Path("dataset/learnext_period_p1")
    test_x_data, test_y_data = load_MeshData(test_dataset_path, style="folders")
    test_dset = FEniCSDataset(test_x_data, test_y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    
    from tools.mesh_quality import mesh_quality_rollout
    mesh_quality_array = mesh_quality_rollout(net, test_dset, 
                                              quality_measure="scaled_jacobian", batch_size=64)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(mesh_quality_array.shape[0]), np.min(mesh_quality_array, axis=1), 'k-', label=r"$\mathcal{N}_\theta$")
    try:
        biharm_mq = np.loadtxt("output/data/biharm_min_mq.csv")
        ax.plot(range(biharm_mq.shape[0]), biharm_mq, 'k:', alpha=0.7, label="biharmonic")
        ax.legend(loc="upper right")
    except:
        pass
    ax.set_xlabel("dataset index (k)")
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlim(xmin=0, xmax=len(test_dset))
    ax.set_ylim(ymin=0.0)
    fig.savefig(results_dir / "min_mesh_mq.pdf")
    fig.savefig(latest_results_dir / "min_mesh_mq.pdf")
    fig.savefig(LOGDIR / LOGENTRY / "min_mesh_mq.pdf")


    if save_xdmf:
        from tools.xdmf_io import pred_to_xdmf

        pred_to_xdmf(net, dset, results_dir / "pred.xdmf", overwrite=xdmf_overwrite)
        shutil.copy(results_dir / "pred.xdmf", latest_results_dir / "pred.xdmf")
        shutil.copy(results_dir / "pred.h5", latest_results_dir / "pred.h5")

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("hyperparameter_study/best_run/problem.json"), type=Path)
    parser.add_argument("--results-dir", default=Path("output/fenics/default"), type=Path)
    parser.add_argument("--save-xdmf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    results_dir: Path = args.results_dir
    save_xdmf: bool = args.save_xdmf
    overwrite: bool = args.overwrite

    if "DL_PASS_OVERWRITE" in os.environ.keys():
        if os.environ["DL_PASS_OVERWRITE"]:
            overwrite = True

    if not problem_file.is_file():
        raise RuntimeError("Given problem description file is not a file.")
    if results_dir.exists():
        if len(list(results_dir.iterdir())) > 0 and overwrite == False:
            print(f"Results directory ({str(results_dir)}) is not empty. Continuing might overwrite data.")
            if not input("Continue? (y/n): ").lower() == "y":
                print("\nExiting program.")
                quit()
    else:
        results_dir.mkdir(parents=True)

    run_boundary_problem(problem_file, results_dir, xdmf_overwrite=True, save_xdmf=save_xdmf)

    return


if __name__ == "__main__":
    main()
