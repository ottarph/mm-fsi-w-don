import argparse
from pathlib import Path


def compute_min_mesh_quality(dset: Path, output: Path) -> None:

    import dolfin as df
    import numpy as np

    from dataset.dataset import load_MeshData
    _, y_data = load_MeshData(dset, style="folders")
    
    msh = y_data.mesh

    from tools.mesh_quality import MeshQuality
    scaled_jacobian = MeshQuality(msh, "scaled_jacobian")
    u = df.Function(df.VectorFunctionSpace(msh, "CG", 1))

    min_mesh_qual_arr = np.zeros(len(y_data))

    for k in range(len(y_data)):
        yh = y_data[k].cpu().numpy()
        u.vector()[:] = yh.reshape(-1)
        min_mesh_qual_arr[k] = np.min(scaled_jacobian(u))

    np.savetxt(output, min_mesh_qual_arr)

    return

def plot_biharm_min_mesh_qual(output: Path, fig_output: Path) -> None:

    import matplotlib.pyplot as plt
    import numpy as np

    min_mesh_qual_arr = np.loadtxt(output)

    fig, ax = plt.subplots()

    ax.plot(range(len(min_mesh_qual_arr)), min_mesh_qual_arr, 'k-')
    ax.set_xlabel("dataset index (k)")
    ax.set_ylabel("scaled Jacobian mesh quality")
    ax.set_xlim(xmin=0, xmax=len(min_mesh_qual_arr))
    ax.set_ylim(ymin=0.0)
    fig.savefig(fig_output)

    return

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    dset: Path = args.dataset
    output: Path = args.output

    output.parent.mkdir(parents=True, exist_ok=True)

    compute_min_mesh_quality(dset, output)

    plot_biharm_min_mesh_qual(output, output.with_suffix(".pdf"))


    return


if __name__ == "__main__":
    main()
