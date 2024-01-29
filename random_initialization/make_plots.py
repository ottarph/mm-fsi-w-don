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

    DATA_DIR = Path("random_initialization/new_study/results/data")
    FIG_DIR = Path("random_initialization/new_study/figures")


    selection = [0, 1, 2, 3, 4, 5, 6,    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    bad_run   = [                     7                                              ]
    full_min_mq_arr, full_val_hist_arr = extract_run_selection(DATA_DIR, range(20))
    selected_min_mq_arr, selected_val_hist_arr = extract_run_selection(DATA_DIR, selection)
    bad_selected_min_mq_arr, bad_selected_val_hist_arr = extract_run_selection("random_initialization/results/data", bad_run)
    bad_selected_min_mq_arr, bad_selected_val_hist_arr = bad_selected_min_mq_arr[0,:], bad_selected_val_hist_arr[0,:]

    fig, axs = make_mesh_quality_quartiles_plot(selected_min_mq_arr)
    fig.savefig(FIG_DIR / "selected_mesh_quality_quartiles.pdf")

    fig, axs = make_mesh_quality_quartiles_plot(selected_min_mq_arr, figsize=(12,5))
    fig.savefig(FIG_DIR / "selected_mesh_quality_quartiles_short.pdf")

    fig, axs = make_conv_quartiles_plot(selected_val_hist_arr)
    fig.savefig(FIG_DIR / "selected_convergence_quartiles.pdf")


    fig, axs = make_single_mesh_quality_plot(bad_selected_min_mq_arr)
    fig.savefig(FIG_DIR / "bad_selected_mesh_quality.pdf")

    fig, axs = make_single_mesh_quality_plot(bad_selected_min_mq_arr, figsize=(12,5))
    fig.savefig(FIG_DIR / "bad_selected_mesh_quality_short.pdf")

    fig, axs = make_single_conv_plot(bad_selected_val_hist_arr)
    fig.savefig(FIG_DIR / "bad_selected_convergence.pdf")

    fig, axs = make_mesh_quality_quartiles_plot(full_min_mq_arr, figsize=(12,5))
    fig.savefig(FIG_DIR / "mq_test.pdf")

    fig, axs = make_conv_quartiles_plot(full_val_hist_arr)
    fig.savefig(FIG_DIR / "conv_test.pdf")

    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fig, axs = make_mesh_quality_quantiles_plot(full_min_mq_arr, qs)
    fig.savefig(FIG_DIR / "mesh_quality_quantiles.pdf")

    fig, axs = make_conv_quantiles_plot(full_val_hist_arr, qs)
    fig.savefig(FIG_DIR / "conv_quantiles.pdf")

    fig, axs = make_mesh_quality_quantiles_plot(full_min_mq_arr, qs, figsize=(12,5))
    fig.savefig(FIG_DIR / "mesh_quality_quantiles_shorter.pdf")

    fig_args = {"figsize": (6.4, 3.2)}
    sp_adj_kws = {"left": 0.1, "bottom": 0.15, "right": 0.97, "top": 0.95}

    fig, axs = make_mesh_quality_quantiles_plot(full_min_mq_arr, qs, **fig_args)
    fig.subplots_adjust(**sp_adj_kws)
    fig.savefig(FIG_DIR / "mesh_quality_quantiles_short.pdf")

    fig, axs = make_conv_quantiles_plot(full_val_hist_arr, qs, **fig_args)
    fig.subplots_adjust(**sp_adj_kws)
    fig.savefig(FIG_DIR / "conv_quantiles_short.pdf")


    try:
        deeponet_min_mq_arr = np.loadtxt("random_initialization/results/data/dno_min_mq.txt")

    except:
        biharm_min_mq = np.loadtxt("output/data/biharm_min_mq.csv")
        noise = np.random.lognormal(mean=-4, sigma=0.8, size=(biharm_min_mq.shape[0], 10))
        deeponet_min_mq_arr = (biharm_min_mq[:,None] - noise).T
        print(f"{deeponet_min_mq_arr.shape = }")

    fig, axs = make_mesh_quality_quartiles_plot(deeponet_min_mq_arr, figsize=(12,5))
    fig.savefig("random_initialization/figures/mq_test.pdf")

    try:
        val_hist_arr = np.loadtxt("random_initialization/results/data/val_hist.txt")

    except:
        T = 40000
        epochs = np.arange(T, dtype=np.float64)
        base_conv = np.loadtxt("results/zeta/data/val.txt")
        noise = np.random.lognormal(mean=-2, sigma=0.5, size=(len(epochs), 10))
        val_hist_arr = (base_conv[:,None] + noise).T
        print(f"{val_hist_arr.shape = }")

    fig, axs = make_conv_quartiles_plot(val_hist_arr)
    fig.savefig("random_initialization/figures/conv_test.pdf")

