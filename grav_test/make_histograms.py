import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
FIGURES_DIR = Path("grav_test/figures")



def make_histograms_shorter_allpos(biharm_smq_arr: np.ndarray, don_smq_arr: np.ndarray) -> tuple[plt.Figure, plt.Axes]:

    custom_bins = np.linspace(0, 1, 51)

    plt.figure()
    fig, axs = plt.subplots(2, 3, figsize=(14,4), sharex=True, sharey=True, layout="constrained")

    ax = axs[0,0]; _ = ax.hist(biharm_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[0,1]; _ = ax.hist(biharm_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[0,2]; _ = ax.hist(biharm_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    
    ax = axs[1,0]; _ = ax.hist(don_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[1,1]; _ = ax.hist(don_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[1,2]; _ = ax.hist(don_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")


    axs[0,0].set_ylabel("biharmonic", fontsize=12)
    axs[1,0].set_ylabel("DeepONet", fontsize=12)

    axs[-1,0].set_xlim(0.0, 1.0)
    axs[-1,1].set_xlim(0.0, 1.0)
    axs[-1,2].set_xlim(0.0, 1.0)


    for k in range(axs.shape[0]):
        axs[k,0].set_ylim(ymin=0.75)


    return fig, axs


if __name__ == "__main__":

    try:
        biharm_signed_mq_arr = np.load("grav_test/data/biharm_signed_mq_arr.npy")
        don_signed_mq_arr = np.load("grav_test/data/don_signed_mq_arr.npy")

    except:
        import grav_test.run_grav_test
        quit()

    fig, axs = make_histograms_shorter_allpos(biharm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms_shorter_allpos.pdf")


