import argparse
import torch
import torch.nn as nn
import shutil

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader

def run_boundary_problem(problem_file: Path, results_dir: Path) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deeponet, dataloader, optimizer, scheduler, loss_fn, num_epochs = load_deeponet_problem(problem_file)

    evaluation_points = dataloader.dataset.y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())


    x0, y0 = next(iter(dataloader))
    z0 = deeponet(x0, evaluation_points)
    loss0 = loss_fn(z0, y0)

    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(nn.Module):
        def __init__(self, deeponet: DeepONet, evaluation_points: torch.Tensor):
            super().__init__()

            self.deeponet = deeponet
            self.register_buffer("evaluation_points", evaluation_points)
            self.evaluation_points: torch.Tensor
            self.evaluation_points.requires_grad_(False)

            return
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.deeponet(x, self.evaluation_points)

    net = EvalWrapper(deeponet, evaluation_points)
    net.to(device)

    context = Context(net, loss_fn, optimizer, scheduler)

    train_with_dataloader(context, dataloader, num_epochs, device, val_dataloader=dataloader)

    context.save_results(results_dir)
    context.save_summary(results_dir)
    context.plot_results(results_dir)
    context.save_model(results_dir)
    shutil.copy(problem_file, results_dir / "problem.json")

    latest_results_dir = Path("results/latest")
    context.save_results(latest_results_dir)
    context.save_summary(latest_results_dir)
    context.plot_results(latest_results_dir)
    shutil.copy(problem_file, latest_results_dir / "problem.json")

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("problems/problem.json"), type=Path)
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
                print("Exiting program.")
                quit()
    else:
        results_dir.mkdir(parents=True)

    run_boundary_problem(problem_file, results_dir)

    return


if __name__ == "__main__":
    main()
