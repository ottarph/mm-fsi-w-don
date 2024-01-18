import dolfin as df
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataset import FEniCSDataset
from tools.translation import torch_to_dolfin

from typing import NewType, Any
Mesh = NewType("Mesh", Any)

class MeshQuality:

    def __init__(self, mesh: Mesh, quality_measure: str = "scaled_jacobian"):

        self.mesh = mesh
        self.quality_measure = quality_measure

        self.polydata = self.build_polydata(mesh)

        return
    
    def build_polydata(self, mesh: df.Mesh) -> pv.PolyData:

        points = np.column_stack((mesh.coordinates()[:,0], mesh.coordinates()[:,1], np.zeros_like(mesh.coordinates()[:,0])))
        faces = np.concatenate((3*np.ones((mesh.num_cells(), 1), dtype=np.uint32), mesh.cells()), axis=1).flatten()

        return pv.PolyData(points, faces)
    
    def convert_vector_field(self, u: df.Function | np.ndarray | torch.Tensor) -> np.ndarray:
        assert isinstance(u, df.Function | np.ndarray | torch.Tensor)

        if isinstance(u, df.Function):
            assert u.function_space().ufl_element().value_shape() == (2,)
            assert u.function_space().mesh() == self.mesh
            uh_tmp = u.compute_vertex_values()
            uh = np.column_stack((uh_tmp[:len(uh_tmp)//2], uh_tmp[len(uh_tmp)//2:], np.zeros(len(uh_tmp)//2)))

        elif isinstance(u, torch.Tensor):
            assert len(u.shape) == 2
            assert u.shape[-1] == 2
            uh_tmp: np.ndarray = u.detach().numpy().astype(np.int64)
            uh = np.column_stack((uh_tmp[:,0], uh_tmp[:,1], np.zeros_like(uh_tmp[:,0])))

        else:
            assert len(u.shape) == 2
            assert u.shape[-1] == 2
            uh = np.column_stack((u[:,0], u[:,1], np.zeros_like(u[:,0])))

        return uh
    
    def __call__(self, u: df.Function | np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Compute the mesh quality of `self.mesh` deformed by u. If `np.ndarray`- or `torch.Tensor`- inputs are used,
        the user must ensure that these are evaluations at mesh vertices in the correct ordering, same as would be
        given by u.compute_vertex_values().

        Args:
            u (df.Function | np.ndarray | torch.Tensor): Function to deform mesh by. If u

        Returns:
            np.ndarray: The mesh quality of all cells in deformed mesh, ordered the same as self.mesh.cells().
        """

        self.polydata["uh"] = self.convert_vector_field(u)
        warped = self.polydata.warp_by_vector("uh")
        quality = warped.compute_cell_quality(quality_measure=self.quality_measure)
        
        return np.copy(quality.cell_data["CellQuality"])


def mesh_quality_rollout(model: nn.Module, dataset: FEniCSDataset, quality_measure: str = "scaled_jacobian",
                         batch_size: int = 1) -> np.ndarray:

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    V = dataset.x_data.function_space
    msh = dataset.x_data.mesh

    mesh_quality = MeshQuality(msh, quality_measure=quality_measure)
    mesh_quality_array = np.zeros((len(dataset), msh.num_cells()), dtype=np.float64)

    u_out = df.Function(V)
    scratch = np.zeros_like(u_out.vector().get_local())

    device = next(iter(model.parameters())).device

    progress_bar = tqdm(total=len(dataset), leave=True)
    k = 0
    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            y = model(x).detach().cpu()
        for b in range(y.shape[0]):
            y_b = y[[b],...]
            torch_to_dolfin(y_b, V, u_out, scratch)
            mesh_quality_array[k,:] = mesh_quality(u_out)
            progress_bar.update(1)
            k += 1

    return mesh_quality_array

def main():

    from pathlib import Path
    import matplotlib.pyplot as plt

    TestFilePath = Path("dataset/learnext_p2/output.xdmf")
    test_file_label = "uh"
    num_test_checkpoints = 207
    test_file = df.XDMFFile(str(TestFilePath))

    msh = df.Mesh()
    test_file.read(msh)

    CG2 = df.VectorFunctionSpace(msh, "CG", 2)
    CG1 = df.VectorFunctionSpace(msh, "CG", 1)
    u_cg2 = df.Function(CG2)
    u_cg1 = df.Function(CG1)

    scaled_jacobian = MeshQuality(msh, "scaled_jacobian")

    from timeit import default_timer as timer
    N = 100
    start = timer()
    for _ in range(N):
        mq = scaled_jacobian(u_cg2)
    end = timer()
    print(f"scaled_jacobian(u_cg2): {(end-start)/N:.2e} s/it")
    N = 100
    start = timer()
    for _ in range(N):
        mq = scaled_jacobian(u_cg1)
    end = timer()
    print(f"scaled_jacobian(u_cg1): {(end-start)/N:.2e} s/it")

    min_mq = np.zeros(num_test_checkpoints)
    for k in tqdm(range(num_test_checkpoints)):
        test_file.read_checkpoint(u_cg2, test_file_label, k)
        mq = scaled_jacobian(u_cg2)
        min_mq[k] = mq.min()

    np.savetxt("output/data/biharm_min_mq.csv", min_mq)
    fig, ax = plt.subplots()
    ax.plot(range(num_test_checkpoints), min_mq, 'k-')
    fig.savefig("output/figures/learnext_p1_biharm_min_mq.pdf")


    return

if __name__ == "__main__":
    main()
