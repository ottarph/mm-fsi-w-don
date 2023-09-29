import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import json

from neuraloperators.loading import load_deeponet_problem

def run_boundary_problem(problem_file: Path, results_dir: Path) -> None:

    deeponet, dataloader, optimizer, scheduler, loss_fn = load_deeponet_problem(problem_file)

    print(deeponet)
    print(dataloader)
    print(optimizer)
    print(scheduler)


    x0, y0 = next(iter(dataloader))
    print(f"{x0.shape = }")
    print(f"{y0.shape = }")
    print(f"{x0.dtype = }")
    print(f"{y0.dtype = }")

    evaluation_points = dataloader.dataset.y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())
    print(f"{evaluation_points.shape = }")
    print(f"{evaluation_points.dtype = }")

    z0 = deeponet(x0, evaluation_points)
    print(f"{z0.shape = }")
    print(f"{z0.dtype = }")

    for k in range(10):
        z0 = deeponet(x0, evaluation_points)
        print(f"{k = }, {loss_fn(z0, y0) = }")
        optimizer.zero_grad()
        loss_fn(z0, y0).backward()
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss_fn(z0, y0).item())
        else:
            scheduler.step()
        print(f"lr = {optimizer.param_groups[0]['lr']}")


    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("problems/problem.json"), type=Path)
    parser.add_argument("--results-dir", default=Path("results/default"), type=Path)
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    results_dir: Path = args.results_dir

    if not problem_file.is_file():
        raise RuntimeError("Given problem description file is not a file.")
    if results_dir.exists():
        if not len(list(results_dir.iterdir())) == 0:
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
