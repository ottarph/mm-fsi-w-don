import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path


plt.rc('axes', labelsize=14)



# DATA_BIH_DIR = Path("../deeponet-learnext-fsi/deeponet_extension/output/run2biharm/data")
DATA_DON_DIR = Path("output/fsi_run_don/data")
DATA_BIH_DIR = Path("output/fsi_run_bih/data")
FIG_DIR = Path("output/figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

drag_comps_don = -np.loadtxt(DATA_DON_DIR / "drag.txt") # Adjust for arbitrary sign
lift_comps_don = -np.loadtxt(DATA_DON_DIR / "lift.txt")
dispy_don = np.loadtxt(DATA_DON_DIR / "displacementy.txt")
determinants_don = np.loadtxt(DATA_DON_DIR / "determinant.txt")
tt_don = np.loadtxt(DATA_DON_DIR / "times.txt")

drag_comps_bih = -np.loadtxt(DATA_BIH_DIR / "drag.txt") # Adjust for arbitrary sign
lift_comps_bih = -np.loadtxt(DATA_BIH_DIR / "lift.txt")
dispy_bih = np.loadtxt(DATA_BIH_DIR / "displacementy.txt")
determinants_bih = np.loadtxt(DATA_BIH_DIR / "determinant.txt")
tt_bih = np.loadtxt(DATA_BIH_DIR / "times.txt")


figsize = (6.4,3.2)
sp_adj_kws = {"top": 0.97, "right": 0.97, "bottom": 0.15}

fig, ax = plt.subplots(figsize=figsize)
fig.subplots_adjust(**sp_adj_kws)
ax.set_xlim(xmin=min(tt_bih[0], tt_don[0]), xmax=max(tt_bih[-1], tt_don[-1]))
ax.plot(tt_bih, lift_comps_bih[:,0] + lift_comps_bih[:,1], 'kx', ms=4, alpha=1.0, label="biharmonic")
ax.plot(tt_don, lift_comps_don[:,0] + lift_comps_don[:,1], 'k-', label="DeepONet")
ax.set_xlabel("$t$")
ax.set_ylabel("lift")
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "lift.pdf")

fig, ax = plt.subplots(figsize=figsize)
fig.subplots_adjust(**sp_adj_kws)
ax.set_xlim(xmin=min(tt_bih[0], tt_don[0]), xmax=max(tt_bih[-1], tt_don[-1]))
ax.plot(tt_bih, drag_comps_bih[:,0] + drag_comps_bih[:,1], 'kx', ms=4, alpha=1.0, label="biharmonic")
ax.plot(tt_don, drag_comps_don[:,0] + drag_comps_don[:,1], 'k-', label="DeepONet")
ax.set_xlabel("$t$")
ax.set_ylabel("drag")
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "drag.pdf")

fig, ax = plt.subplots(figsize=figsize)
fig.subplots_adjust(**sp_adj_kws)
ax.set_xlim(xmin=min(tt_bih[0], tt_don[0]), xmax=max(tt_bih[-1], tt_don[-1]))
ax.plot(tt_bih, dispy_bih, 'kx', ms=4, alpha=1.0, label="biharmonic")
ax.plot(tt_don, dispy_don, 'k-', label="DeepONet")
ax.set_yticks([-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08])
ax.set_xlabel("$t$")
ax.set_ylabel("$y$-displacement")
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "dispy.pdf")

fig, ax = plt.subplots(figsize=figsize)
fig.subplots_adjust(**sp_adj_kws)
ax.set_xlim(xmin=min(tt_bih[0], tt_don[0]), xmax=max(tt_bih[-1], tt_don[-1]))
ax.plot(tt_bih, determinants_bih, 'kx', ms=4, alpha=1.0, label="biharmonic")
ax.plot(tt_don, determinants_don, 'k-', label="DeepONet")
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\mathrm{min} \; J$")
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "determinant.pdf")
