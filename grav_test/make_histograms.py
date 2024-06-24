import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
FIGURES_DIR = Path("grav_test/figures")

def make_histograms(biharm_smq_arr: np.ndarray, harm_smq_arr: np.ndarray, don_smq_arr: np.ndarray) -> tuple[plt.Figure, plt.Axes]:

    custom_bins = list(np.linspace(-1, 0, 11)[:-1]) + list(np.linspace(0, 1, 41))

    plt.figure()
    fig, axs = plt.subplots(4, 3, figsize=(14,8), sharex=True, sharey=True, layout="constrained")

    ax = axs[1,0]; _ = ax.hist(biharm_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,1]; _ = ax.hist(biharm_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,2]; _ = ax.hist(biharm_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[2,0]; _ = ax.hist(harm_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Harmonic")
    ax = axs[2,1]; _ = ax.hist(harm_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Harmonic")
    ax = axs[2,2]; _ = ax.hist(harm_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Harmonic")
    ax = axs[3,0]; _ = ax.hist(don_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[3,1]; _ = ax.hist(don_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[3,2]; _ = ax.hist(don_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")


    axs[1,0].set_ylabel("biharmonic", fontsize=12)
    axs[2,0].set_ylabel("harmonic", fontsize=12)
    axs[3,0].set_ylabel("DeepONet", fontsize=12)

    axs[-1,0].set_xlim(-1.0, 1.0)
    axs[-1,1].set_xlim(-1.0, 1.0)
    axs[-1,2].set_xlim(-1.0, 1.0)

    [axs[k,l].axvline(x=0.0, color="black", lw=0.8, alpha=0.8) for k in range(1,axs.shape[0]) for l in range(axs.shape[1])]

    def forward(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 0.2*x, x)
    def inverse(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 5.0*x, x)

    for k in range(axs.shape[0]):
        for l in range(axs.shape[1]):
            scale = mpl.scale.FuncScale(axs[k,l], (forward, inverse))
            axs[k,l].set_xscale(scale)
            xticks = np.array([-1.0, -0.5]+list(np.linspace(0, 1, 6)))
            axs[k,l].set_xticks(xticks)
            axs[k,l].set_xticklabels([np.format_float_positional(xi, precision=2) for xi in xticks])
        axs[k,0].set_ylim(ymin=0.75)


    for l in range(axs.shape[1]):
        axs[0,l].remove()

    ax = fig.add_subplot(4, 3, 1)
    img = plt.imread(FIGURES_DIR / "def_1_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(4, 3, 2)
    img = plt.imread(FIGURES_DIR / "def_2_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(4, 3, 3)
    img = plt.imread(FIGURES_DIR / "def_3_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()

    return fig, axs



def make_histograms_short(biharm_smq_arr: np.ndarray, don_smq_arr: np.ndarray) -> tuple[plt.Figure, plt.Axes]:

    custom_bins = list(np.linspace(-1, 0, 11)[:-1]) + list(np.linspace(0, 1, 41))

    plt.figure()
    fig, axs = plt.subplots(3, 3, figsize=(14,6), sharex=True, sharey=True, layout="constrained")

    ax = axs[1,0]; _ = ax.hist(biharm_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,1]; _ = ax.hist(biharm_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,2]; _ = ax.hist(biharm_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    
    ax = axs[2,0]; _ = ax.hist(don_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[2,1]; _ = ax.hist(don_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[2,2]; _ = ax.hist(don_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")


    axs[1,0].set_ylabel("biharmonic", fontsize=12)
    axs[2,0].set_ylabel("DeepONet", fontsize=12)

    axs[-1,0].set_xlim(-1.0, 1.0)
    axs[-1,1].set_xlim(-1.0, 1.0)
    axs[-1,2].set_xlim(-1.0, 1.0)

    [axs[k,l].axvline(x=0.0, color="black", lw=0.8, alpha=0.8) for k in range(1,axs.shape[0]) for l in range(axs.shape[1])]

    def forward(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 0.2*x, x)
    def inverse(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 5.0*x, x)

    for k in range(axs.shape[0]):
        for l in range(axs.shape[1]):
            scale = mpl.scale.FuncScale(axs[k,l], (forward, inverse))
            axs[k,l].set_xscale(scale)
            xticks = np.array([-1.0, -0.5]+list(np.linspace(0, 1, 6)))
            axs[k,l].set_xticks(xticks)
            axs[k,l].set_xticklabels([np.format_float_positional(xi, precision=2) for xi in xticks])
        axs[k,0].set_ylim(ymin=0.75)


    for l in range(axs.shape[1]):
        axs[0,l].remove()

    # add_subplot_kws = {"layout": "constrained"}
    add_subplot_kws = {"sharex": axs[1,0]}#, "sharey": axs[1,0]}
    ax = fig.add_subplot(3, 3, 1)#, sharex=axs[1,0])
    img = plt.imread(FIGURES_DIR / "def_1_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(3, 3, 2)#, sharex=axs[1,1])
    img = plt.imread(FIGURES_DIR / "def_2_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(3, 3, 3)#, sharex=axs[1,2])
    img = plt.imread(FIGURES_DIR / "def_3_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()

    return fig, axs




def make_histograms_shorter(biharm_smq_arr: np.ndarray, don_smq_arr: np.ndarray) -> tuple[plt.Figure, plt.Axes]:

    custom_bins = list(np.linspace(-1, 0, 11)[:-1]) + list(np.linspace(0, 1, 41))

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

    axs[-1,0].set_xlim(-1.0, 1.0)
    axs[-1,1].set_xlim(-1.0, 1.0)
    axs[-1,2].set_xlim(-1.0, 1.0)

    [axs[k,l].axvline(x=0.0, color="black", lw=0.8, alpha=0.8) for k in range(axs.shape[0]) for l in range(axs.shape[1])]

    def forward(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 0.2*x, x)
    def inverse(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, 5.0*x, x)

    for k in range(axs.shape[0]):
        for l in range(axs.shape[1]):
            scale = mpl.scale.FuncScale(axs[k,l], (forward, inverse))
            axs[k,l].set_xscale(scale)
            xticks = np.array([-1.0, -0.5]+list(np.linspace(0, 1, 6)))
            axs[k,l].set_xticks(xticks)
            axs[k,l].set_xticklabels([np.format_float_positional(xi, precision=2) for xi in xticks])
        axs[k,0].set_ylim(ymin=0.75)


    return fig, axs



def make_histograms_short_allpos(biharm_smq_arr: np.ndarray, don_smq_arr: np.ndarray) -> tuple[plt.Figure, plt.Axes]:

    custom_bins = np.linspace(0, 1, 51)

    plt.figure()
    fig, axs = plt.subplots(3, 3, figsize=(14,6), sharex=True, sharey=True, layout="constrained")

    ax = axs[1,0]; _ = ax.hist(biharm_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,1]; _ = ax.hist(biharm_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    ax = axs[1,2]; _ = ax.hist(biharm_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"Biharmonic")
    
    ax = axs[2,0]; _ = ax.hist(don_smq_arr[0,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[2,1]; _ = ax.hist(don_smq_arr[1,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")
    ax = axs[2,2]; _ = ax.hist(don_smq_arr[2,:], color="#363533", bins=custom_bins, density=False); ax.set_yscale("log"); # ax.set_title(f"DeepONet")


    axs[1,0].set_ylabel("biharmonic", fontsize=12)
    axs[2,0].set_ylabel("DeepONet", fontsize=12)

    axs[-1,0].set_xlim(0.0, 1.0)
    axs[-1,1].set_xlim(0.0, 1.0)
    axs[-1,2].set_xlim(0.0, 1.0)


    for k in range(axs.shape[0]):
        axs[k,0].set_ylim(ymin=0.75)


    for l in range(axs.shape[1]):
        axs[0,l].remove()

    # add_subplot_kws = {"layout": "constrained"}
    add_subplot_kws = {"sharex": axs[1,0]}#, "sharey": axs[1,0]}
    ax = fig.add_subplot(3, 3, 1)#, sharex=axs[1,0])
    img = plt.imread(FIGURES_DIR / "def_1_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(3, 3, 2)#, sharex=axs[1,1])
    img = plt.imread(FIGURES_DIR / "def_2_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()
    ax = fig.add_subplot(3, 3, 3)#, sharex=axs[1,2])
    img = plt.imread(FIGURES_DIR / "def_3_soft_blk.png")
    _ = ax.imshow(img)
    ax.set_axis_off()

    return fig, axs




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
        harm_signed_mq_arr = np.load("grav_test/data/harm_signed_mq_arr.npy")
        biharm_signed_mq_arr = np.load("grav_test/data/biharm_signed_mq_arr.npy")
        don_signed_mq_arr = np.load("grav_test/data/don_signed_mq_arr.npy")

    except:
        import grav_test.run_grav_test
        quit()

    fig, axs = make_histograms(biharm_signed_mq_arr, harm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms.pdf")

    fig, axs = make_histograms_short(biharm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms_short.pdf")

    fig, axs = make_histograms_shorter(biharm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms_shorter.pdf")

    fig, axs = make_histograms_short_allpos(biharm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms_short_allpos.pdf")

    fig, axs = make_histograms_shorter_allpos(biharm_signed_mq_arr, don_signed_mq_arr)
    fig.savefig("grav_test/figures/grav_test_histograms_shorter_allpos.pdf")


