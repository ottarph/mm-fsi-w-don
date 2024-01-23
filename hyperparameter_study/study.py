import torch
import torch.nn as nn
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import copy
import timeit
import datetime

from pathlib import Path
from tools.mesh_quality import mesh_quality_rollout

from neuraloperators.deeponet import DeepONet
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader, save_model
from dataset.dataset import MeshData, FEniCSDataset, load_MeshData, ToDType

def run_boundary_problem_hpar(problem_file: Path, results_dir: Path) -> tuple[DeepONet, float]:

    latest_results_dir: Path = Path("hyperparameter_study/latest_results")
    device = "cuda"

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
    
    show_mb_pbar = False

    try:
        train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader=val_dataloader,
                              show_minibatch_pbar=show_mb_pbar)
    except KeyboardInterrupt:
        if input("Training interrupted. Save current progress? (y/n): ").lower() != "y":
            quit()

    context.save_summary(results_dir)
    histfig = context.plot_results(results_dir)
    plt.close(histfig)
    shutil.copy(problem_file, results_dir / "problem.json")

    context.save_summary(latest_results_dir)
    histfig = context.plot_results(latest_results_dir)
    plt.close(histfig)
    shutil.copy(problem_file, latest_results_dir / "problem.json")


    N_timer = 2000
    x0, y0 = next(iter(train_dataloader))
    x0 = x0[0]
    x0_c = x0.to("cuda")
    gpu_time = timeit.timeit(lambda: deeponet(x0_c, x0_c), number=N_timer) / N_timer

    net.to("cpu")
    deeponet.to("cpu")

    cpu_time = timeit.timeit(lambda: deeponet(x0, x0), number=N_timer) / N_timer

    print(f"{cpu_time = :.2e}, {gpu_time = :.2e}")


    test_dataset_path = Path("dataset/learnext_period_p1")
    test_x_data, test_y_data = load_MeshData(test_dataset_path, style="folders")
    test_dset = FEniCSDataset(test_x_data, test_y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    
    
    mesh_quality_array = mesh_quality_rollout(net, test_dset, 
                                              quality_measure="scaled_jacobian", batch_size=64)
    deeponet_min_mq = mesh_quality_array.min(axis=1)
    deeponet_min_mq = np.min(mesh_quality_array, axis=1)
    biharm_min_mq = np.loadtxt("output/data/biharm_min_mq.csv")
    diff_mq = biharm_min_mq - deeponet_min_mq
    max_diff_mq = np.max(diff_mq)
    

    datarr = np.array([max_diff_mq, cpu_time, gpu_time])[None,:]
    np.savetxt(results_dir / "data.txt", datarr, header="max_diff_mq, cpu_time, gpu_time")
    np.savetxt(latest_results_dir / "data.txt", datarr, header="max_diff_mq, cpu_time, gpu_time")

    fig, ax = plt.subplots()
    ax.plot(range(deeponet_min_mq.shape[0]), deeponet_min_mq, 'k-', label=r"DeepONet")
    ax.plot(range(biharm_min_mq.shape[0]), biharm_min_mq, 'k:', alpha=0.7, label="biharmonic")

    ax.legend(loc="upper right")
    ax.set_xlabel("dataset index (k)")
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlim(xmin=0, xmax=len(test_dset))
    ax.set_ylim(ymin=0.0)
    fig.savefig(results_dir / "min_mesh_mq.pdf")
    fig.savefig(latest_results_dir / "min_mesh_mq.pdf")

    plt.close(fig)

    return deeponet, max_diff_mq



def run_problem_conf(problem_dict: dict, SAVE_TO_DIR: Path, config_dict: dict,
                     best_max_diff_mq: float):

    SAVE_TO_DIR.mkdir(parents=True, exist_ok=True)

    STAGING_DIR = Path(config_dict["STAGING_DIR"])
    (STAGING_DIR / "problem.json").unlink(missing_ok=True)

    BEST_RUN_DIR = Path(config_dict["BEST_RUN_DIR"])

    with open(STAGING_DIR / "problem.json", "w") as outfile:
        json.dump(problem_dict, outfile, indent=4)
        
    deeponet, max_diff_mq = run_boundary_problem_hpar(problem_file=STAGING_DIR / "problem.json",
                              results_dir=SAVE_TO_DIR)

    with open(SAVE_TO_DIR / "problem.json", "r") as infile:
        new_dict = json.load(infile)
    assert problem_dict == new_dict

    if max_diff_mq < best_max_diff_mq:
        shutil.copytree(SAVE_TO_DIR, BEST_RUN_DIR, dirs_exist_ok=True)
        save_model(deeponet.trunk, BEST_RUN_DIR / "trunk.pt")
        save_model(deeponet.branch, BEST_RUN_DIR / "branch.pt")
        (BEST_RUN_DIR / "run.txt").write_text(
            str(SAVE_TO_DIR.parts[-2])+"/"+str(SAVE_TO_DIR.parts[-1])
        )
        best_max_diff_mq = max_diff_mq


    return best_max_diff_mq

def main():

    CONFIG_PATH = "hyperparameter_study/study_config.json"
    BASE_PROBLEM_FILE_PATH = "hyperparameter_study/base_problem.json"

    with open(CONFIG_PATH, "r") as infile:
        config_dict = json.load(infile)

    with open(BASE_PROBLEM_FILE_PATH, "r") as infile:
        base_problem_dict = json.load(infile)

    num_combinations = len(config_dict["widths"]) * len(config_dict["depths"]) * len(config_dict["basis_sizes"])
    num_runs = num_combinations * config_dict["runs_per_combination"]
    study_time = num_runs * config_dict["expected_time_per_run"]


    print()

    widths = config_dict["widths"]
    depths = config_dict["depths"]
    basis_sizes = config_dict["basis_sizes"]
    runs_per_combination = config_dict["runs_per_combination"]
    epochs_per_run = config_dict["epochs_per_run"]
    expected_time_per_epoch = config_dict["expected_time_per_epoch"]
    expected_time_per_run = expected_time_per_epoch * epochs_per_run
    num_runs = num_combinations * runs_per_combination
    study_time = num_runs * expected_time_per_run
    
    print(f"{num_combinations = }")
    print(f"{runs_per_combination = }")
    print(f"{num_runs = }")
    print(f"{expected_time_per_epoch = }")
    print(f"{expected_time_per_run = }")
    print(f"study_time = {str(datetime.timedelta(seconds=study_time))}")
    
    print()

    RESULTS_DIR = Path(config_dict["RESULTS_DIR"])
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    LOG_FILE_PATH = Path(config_dict["LOG_FILE"])

    assert len(widths) < 10 and len(depths) < 10 and len(basis_sizes) < 10 and runs_per_combination < 26

    current_run = 1

    best_max_diff_mq = np.inf
    for i, w in enumerate(widths):
        for j, d in enumerate(depths):
            for k, p in enumerate(basis_sizes):
                for run in range(runs_per_combination):
                    CONF_DIR = RESULTS_DIR / f"{i}{j}{k}"
                    RUN_DIR = CONF_DIR / "abcdefghijklmnopqrstuvwxyz"[run]
                    RUN_DIR.mkdir(exist_ok=True, parents=True)
                    problem_dict = copy.deepcopy(base_problem_dict)
                    problem_dict["branch"]["MLP"]["widths"] = [412] + [w] * d + [p]
                    problem_dict["trunk"]["MLP"]["widths"] = [2] + [w] * d + [p]
                    problem_dict["seed"] = run
                    problem_dict["num_epochs"] = epochs_per_run
                    print()
                    print(f"{w = }, {d = }, {p = }, {run = }, progress {current_run} / {num_runs}")
                    with open(LOG_FILE_PATH, "a") as outfile:
                        outfile.write(
    f"{datetime.datetime.now().strftime(r'%Y_%m_%d-%H_%M')}: {w = }, {d = }, {p = }, {run = }, progress {current_run} / {num_runs}\n\n"
                        )

                    best_max_diff_mq = run_problem_conf(problem_dict, RUN_DIR, 
                                                        config_dict, best_max_diff_mq)
                    current_run += 1

    with open(LOG_FILE_PATH, "a") as outfile:
        outfile.write(
            f"{datetime.datetime.now().strftime(r'%Y_%m_%d-%H_%M')}: Finish.\n\n"
                     )

    return

if __name__ == "__main__":
    main()
