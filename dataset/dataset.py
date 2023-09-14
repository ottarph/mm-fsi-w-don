import torch
import numpy as np
import dolfin as df

from torch.utils.data import Dataset
from torch.nn import Module, Identity
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal


class MeshData:

    def __init__(self, mesh: df.Mesh, function_space: df.FunctionSpace, 
                 checkpoints: Iterable[int]):
        
        self.mesh = mesh
        self.function_space = function_space
        self.checkpoints = checkpoints

        return

    def __len__(self) -> int:
        return len(self.checkpoints)

    @property
    def boundary_dofs(self) -> np.ndarray:

        if not hasattr(self, "boundary_dof_indices"):

            V = df.FunctionSpace(self.mesh, "CG", self.function_space.ufl_element().degree())
            u = df.Function(V)
            bc = df.DirichletBC(V, df.Constant(1), "on_boundary")

            u.vector()[:] = 0.0
            bc.apply(u.vector())

            ids = np.flatnonzero(u.vector().get_local()).astype(np.int32)

            self.boundary_dof_indices = ids

        return self.boundary_dof_indices
    

class MeshDataXDMF(MeshData):

    def __init__(self, filename: PathLike, type: Literal["scalar", "vector", "tensor"], 
                 dim: int, degree: int, checkpoints: Iterable[int], function_label: str = "uh"):
        
        self.filename = Path(filename)
        self.file = df.XDMFFile(str(filename))
        self.type = type
        self.dim = dim
        self.degree = degree
        self.checkpoints = checkpoints
        self.function_label = function_label

        self.mesh = df.Mesh()
        self.file.read(self.mesh)
        assert self.mesh.num_vertices() > 0

        if type == "scalar":
            raise NotImplementedError()
        elif type == "vector":
            self.function_space = df.VectorFunctionSpace(self.mesh, "CG", degree, dim)
        elif type == "tensor":
            raise NotImplementedError()
        else:
            raise ValueError()
        
        self.function = df.Function(self.function_space)

        super().__init__(self.mesh, self.function_space, self.checkpoints)

        return

    
    def __getitem__(self, index: int) -> torch.Tensor:

        if self.type == "scalar":
            raise NotImplementedError
        if self.type == "tensor":
            raise NotImplementedError
        
        self.file.read_checkpoint(self.function, self.function_label, index)

        uh0 = self.function.vector().get_local()
        """
            Memory order: [u_x_0, u_y_0, u_x_1, u_y_1, ..., u_x_N, u_y_N]
        """

        uh = np.zeros((uh0.shape[0] // self.dim, self.dim), dtype=uh0.dtype)
        for k in range(self.dim):
            uh[:,k] = uh0[k::self.dim]
        """
            Memory order:
            [[u_x_0, u_y_0]
             [u_x_1, u_y_1]
                   ...
             [u_x_N, u_y_N]]
        """

        return torch.tensor(uh)

 
class MeshDataFolders(MeshData):

    def __init__(self, directory: PathLike, filename: PathLike,
                 checkpoints: Iterable[int], 
                 type: Literal["scalar", "vector", "tensor"], degree: int, dim: int,
                 num_digits: int = -1):
        """
            `directory` is the folder where .npy-files are stored,
            `file` is the .xdmf-file that was converted.
        """
        self.directory = Path(directory)
        self.filename = Path(filename)
        self.file = df.XDMFFile(str(filename))

        self.checkpoints = checkpoints

        if num_digits == -1:
            num_digits = int(np.floor(np.log10(len(checkpoints))))+1
        self.num_digits = num_digits

        self.mesh = df.Mesh()
        self.file.read(self.mesh)
        assert self.mesh.num_vertices() > 0

        if type == "scalar":
            raise NotImplementedError()
        elif type == "vector":
            self.function_space = df.VectorFunctionSpace(self.mesh, "CG", degree, dim)
        elif type == "tensor":
            raise NotImplementedError()
        else:
            raise ValueError()

        super().__init__(self.mesh, self.function_space, self.checkpoints)

        return
    
    def __getitem__(self, index: int) -> torch.Tensor:

        load_name = f"{index:0{self.num_digits}d}.npy"
        arr = np.load(self.directory / load_name)

        return torch.tensor(arr)
    

class FEniCSDataset(Dataset):

    def __init__(self, x_data: MeshData, y_data: MeshData,
                       x_transform: Module = Identity(), 
                       y_transform: Module = Identity()):
        super().__init__()

        self.x_data = x_data
        self.y_data = y_data

        self.x_transform = x_transform
        self.y_transform = y_transform

        assert x_data.checkpoints == y_data.checkpoints
        self.checkpoints = x_data.checkpoints

        return
    
    def __len__(self) -> int:
        return len(self.checkpoints)
    
    def __getitem__(self, index: int) -> torch.Tensor:

        x = self.x_data[index]
        y = self.y_data[index]
        
        x = self.x_transform(x)
        y = self.y_transform(y)

        return x, y
    

def load_MeshData(directory: PathLike, style: Literal["XDMF", "folders"] = "folders") -> tuple[MeshData, MeshData]:
    directory = Path(directory)
    import json

    with open(directory / "info.json", "r") as infile:
        info_dict= json.loads(infile.read())

    if not ( (directory / "input.xdmf").exists() and (directory / "output.xdmf").exists() ):
        raise FileNotFoundError("Can not find dataset .xdmf-files")

    input_dict = info_dict["input"]
    output_dict = info_dict["output"]

    if style == "XDMF":

        x_data = MeshDataXDMF(directory / "input.xdmf", input_dict["type"],
                              input_dict["dim"], input_dict["degree"], 
                              range(info_dict["num_checkpoints"]), input_dict["label"])
        y_data = MeshDataXDMF(directory / "output.xdmf", output_dict["type"],
                              output_dict["dim"], output_dict["degree"], 
                              range(info_dict["num_checkpoints"]), output_dict["label"])

    elif style == "folders":

        if not ( (directory / "input_dir").exists() and (directory / "output_dir").exists() ):
            raise FileNotFoundError("Can not find dataset folders")
        
        x_data = MeshDataFolders(directory / "input_dir", directory / "input.xdmf",
                                 range(info_dict["num_checkpoints"]), input_dict["type"], 
                                 input_dict["degree"], input_dict["dim"])
        y_data = MeshDataFolders(directory / "output_dir", directory / "output.xdmf",
                                 range(info_dict["num_checkpoints"]), output_dict["type"], 
                                 output_dict["degree"], output_dict["dim"])
    
    else:
        raise ValueError()

    return x_data, y_data


class OnBoundary(Module):

    def __init__(self, data: MeshData):
        super().__init__()

        self.data = data
        self.indices = torch.tensor(data.boundary_dofs)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.index_select(x, -2, self.indices)


class AddCoordinates(Module):

    def __init__(self, data: MeshData):
        super().__init__()

        raise NotImplementedError()    
