import dolfin as df
import numpy as np


from os import PathLike
from pathlib import Path
from typing import Callable
from tqdm import tqdm

def make_displacements(solid_mesh_path: PathLike, surface_load_func: Callable[[float], float], save_path: PathLike) -> None:
    solid_mesh_path = Path(solid_mesh_path)
    save_path = Path(save_path)

    if save_path.exists():
        save_path.unlink()

    solid_mesh = df.Mesh()
    with df.HDF5File(solid_mesh.mpi_comm(), str(solid_mesh_path), 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with df.HDF5File(solid_mesh.mpi_comm(), str(solid_mesh_path), 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')


    out_file = df.XDMFFile(str(save_path))

    # ----6----
    # 4       9
    # ----6----

    displacement_bcs = {4: df.Constant((0.0, 0.0))}
    volume_load = df.Constant((0.0, 0.0))

    from artificial_data.elasticity import solve_neohook_solid
    
    for k, theta in enumerate(tqdm(np.linspace(0, 2*np.pi, 101), leave=False)):
        surface_load = surface_load_func(theta)
        uh = solve_neohook_solid(boundaries=solid_boundaries,
                                    volume_load=volume_load,
                                    surface_load=surface_load,
                                    displacement_bcs=displacement_bcs,
                                    mu=df.Constant(0.5e6),
                                    lmbda=df.Constant(2.0e6),
                                    pdegree=2)


        out_file.write_checkpoint(uh, "uh", k, append=True)
