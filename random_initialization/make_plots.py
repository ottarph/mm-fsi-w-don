import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from os import PathLike
from pathlib import Path
from typing import Sequence



def extract_run_selection(DATA_DIR: PathLike, runs: Sequence[int]) -> np.ndarray:

    DATA_DIR = Path(DATA_DIR)

    min_mq_arr = np.loadtxt(DATA_DIR / "dno_min_mq.txt")
    selected_min_mq_arr = np.copy(min_mq_arr[runs,:])

    val_hist_arr = np.loadtxt(DATA_DIR / "val_hist.txt")
    selected_val_hist_arr = np.copy(val_hist_arr[runs,:])

    return selected_min_mq_arr, selected_val_hist_arr


def make_mesh_quality_quantiles_plot(deeponet_min_mq_arr: np.ndarray, qs: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:

    assert len(qs) == 5

    biharm_min_mq = np.loadtxt("output/data/biharm_min_mq.csv")

    quantiles_mesh_mqs = np.quantile(deeponet_min_mq_arr, qs, axis=0)
    
    kk = range(quantiles_mesh_mqs.shape[1])

    fig, ax = plt.subplots(**fig_args)

    ax.plot(kk, biharm_min_mq, color="black", linestyle="dotted", label="biharmonic", alpha=1.0, lw=1.0)

    ax.plot(kk, quantiles_mesh_mqs[2], color="black", linestyle="solid", label="median")
    ax.fill_between(kk, quantiles_mesh_mqs[0], quantiles_mesh_mqs[4], color="black", alpha=0.12, label=f"{qs[0]:.2f}-{qs[-1]:.2f} quantiles")
    ax.fill_between(kk, quantiles_mesh_mqs[1], quantiles_mesh_mqs[3], color="black", alpha=0.3,  label=f"{qs[1]:.2f}-{qs[-2]:.2f} quantiles")

    ax.set_xlim(xmin=0, xmax=kk[-1])
    ax.set_ylim(ymin=0.0)
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlabel("dataset index ($k$)")
    ax.legend()


    return fig, ax


def make_conv_quantiles_plot(val_hist_arr: np.ndarray, qs: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:

    assert len(qs) == 5

    fig, ax = plt.subplots(**fig_args)

    quantiles_mesh_mqs = np.quantile(val_hist_arr, qs, axis=0)
    
    epochs = np.arange(val_hist_arr.shape[1], dtype=np.float64)

    ax.plot(epochs, quantiles_mesh_mqs[2], color="black", linestyle="solid", label="median")
    ax.fill_between(epochs, quantiles_mesh_mqs[0], quantiles_mesh_mqs[4], color="black", alpha=0.12, label=f"{qs[0]:.2f}-{qs[-1]:.2f} quantiles")
    ax.fill_between(epochs, quantiles_mesh_mqs[1], quantiles_mesh_mqs[3], color="black", alpha=0.3,  label=f"{qs[1]:.2f}-{qs[-2]:.2f} quantiles")

    ax.set_xlim(xmin=0, xmax=epochs[-1])
    ax.set_ylabel("validation loss")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.legend()

    return fig, ax


def make_mesh_quality_quartiles_plot(deeponet_min_mq_arr: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:


    biharm_min_mq = np.loadtxt("output/data/biharm_min_mq.csv")

    qs = np.array([0, 0.25, 0.5, 0.75, 1.0])
    quartiles_mesh_mqs = np.quantile(deeponet_min_mq_arr, qs, axis=0)
    
    kk = range(quartiles_mesh_mqs.shape[1])

    fig, ax = plt.subplots(**fig_args)

    ax.plot(kk, biharm_min_mq, color="black", linestyle="dotted", label="biharmonic", alpha=1.0, lw=1.0)

    ax.plot(kk, quartiles_mesh_mqs[2], color="black", linestyle="solid", label="median")
    ax.fill_between(kk, quartiles_mesh_mqs[0], quartiles_mesh_mqs[4], color="black", alpha=0.12, label="0.00-1.00 quantiles")
    ax.fill_between(kk, quartiles_mesh_mqs[1], quartiles_mesh_mqs[3], color="black", alpha=0.3, label="0.25-0.75 quantiles")

    ax.set_xlim(xmin=0, xmax=kk[-1])
    ax.set_ylim(ymin=0.0)
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlabel("dataset index ($k$)")
    ax.legend()


    return fig, ax


def make_conv_quartiles_plot(val_hist_arr: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:


    fig, ax = plt.subplots(**fig_args)

    qs = np.array([0, 0.25, 0.5, 0.75, 1.0])
    quartiles_mesh_mqs = np.quantile(val_hist_arr, qs, axis=0)
    
    epochs = np.arange(val_hist_arr.shape[1], dtype=np.float64)

    ax.plot(epochs, quartiles_mesh_mqs[2], color="black", linestyle="solid", label="median")
    ax.fill_between(epochs, quartiles_mesh_mqs[0], quartiles_mesh_mqs[4], color="black", alpha=0.12, label="0.00-1.00 quantiles")
    ax.fill_between(epochs, quartiles_mesh_mqs[1], quartiles_mesh_mqs[3], color="black", alpha=0.3, label="0.25-0.75 quantiles")

    ax.set_xlim(xmin=0, xmax=epochs[-1])
    ax.set_ylabel("validation loss")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.legend()

    return fig, ax


def make_single_mesh_quality_plot(deeponet_min_mq_arr: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:

    assert len(deeponet_min_mq_arr.shape) == 1

    biharm_min_mq = np.loadtxt("output/data/biharm_min_mq.csv")
    
    kk = range(deeponet_min_mq_arr.shape[0])

    fig, ax = plt.subplots(**fig_args)

    ax.plot(kk, biharm_min_mq, color="black", linestyle="dotted", label="biharmonic", alpha=1.0, lw=1.0)
    ax.plot(kk, deeponet_min_mq_arr, color="black", linestyle="solid", label="DeepONet")

    ax.set_xlim(xmin=0, xmax=kk[-1])
    ax.set_ylim(ymin=0.0)
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlabel("dataset index ($k$)")
    ax.legend()

    return fig, ax


def make_single_conv_plot(val_hist_arr: np.ndarray, **fig_args) -> tuple[plt.Figure, plt.Axes]:

    assert len(val_hist_arr.shape) == 1

    fig, ax = plt.subplots(**fig_args)
    
    epochs = np.arange(val_hist_arr.shape[0], dtype=np.float64)

    ax.plot(epochs, val_hist_arr, color="black", linestyle="solid")

    ax.set_xlim(xmin=0, xmax=epochs[-1])
    ax.set_ylabel("validation loss")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")

    return fig, ax


if __name__ == "__main__":
    import json
    
    CONFIG_PATH = "random_initialization/study_config.json"
    with open(CONFIG_PATH, "r") as infile:
        config_dict = json.load(infile)
    DATA_DIR = Path(config_dict["DATA_DIR"])
    FIG_DIR = Path("output/figures")

    NUM_RUNS = config_dict["num_runs"]

    min_mq_arr, val_hist_arr = extract_run_selection(DATA_DIR, range(NUM_RUNS))


    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    fig_args = {"figsize": (6.4, 3.2)}
    sp_adj_kws = {"left": 0.1, "bottom": 0.15, "right": 0.97, "top": 0.95}

    fig, axs = make_mesh_quality_quantiles_plot(min_mq_arr, qs, **fig_args)
    fig.subplots_adjust(**sp_adj_kws)
    fig.savefig(FIG_DIR / "random_init_mesh_quality_quantiles.pdf")

    fig, axs = make_conv_quantiles_plot(val_hist_arr, qs, **fig_args)
    fig.subplots_adjust(**sp_adj_kws)
    fig.savefig(FIG_DIR / "random_init_conv_quantiles.pdf")


