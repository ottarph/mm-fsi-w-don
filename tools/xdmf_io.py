import dolfin as df
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from dataset.dataset import FEniCSDataset, MeshData
from os import PathLike
from pathlib import Path
from tqdm import tqdm


def pred_checkpoint_dolfin_to_dolfin(model: nn.Module, uh: df.Function, 
                    V: df.FunctionSpace, u_out: df.Function | None = None) -> df.Function:
    """ Gives dolfin-function representation of neural network prediction of one checkpoint of dataset, 
        represented as dolfin-function.
        Assumes input and output function spaces are the same. """
    
    if u_out is None:
        u_out = df.Function(V)

    if len(V.ufl_element().value_shape()) == 0:
        """ Scalar element """
        raise NotImplementedError()
    
    elif len(V.ufl_element().value_shape()) == 1:
        """ Vector element """

        with torch.no_grad():
            uh_np = uh.vector().get_local()
            uh_reshape = np.zeros((uh_np.shape[0] // V.ufl_element().value_size(), *V.ufl_element().value_shape()))

            for d in range(uh_reshape.shape[1]):
                uh_reshape[:,d] = uh_np[d::uh_reshape.shape[1]]

            uh_torch = torch.tensor(uh_reshape, dtype=torch.get_default_dtype())
            pred = model(uh_torch)
            uh_reshape = pred.detach().numpy()
            
            for d in range(uh_reshape.shape[1]):
                uh_np[d::uh_reshape.shape[1]] = uh_reshape[:,d]

            u_out.vector().set_local(uh_np)

    else:
        """ Tensor element """
        raise NotImplementedError()

    return u_out

def pred_checkpoint_torch_to_dolfin(model: nn.Module, uh: torch.Tensor, 
                    V: df.FunctionSpace, u_out: df.Function | None = None) -> df.Function:
    """ Gives dolfin-function representation of neural network prediction of one checkpoint of dataset, 
        represented as torch-tensor.
        Assumes input and output function spaces are the same. """
    

    if u_out is None:
        u_out = df.Function(V)

    if len(V.ufl_element().value_shape()) == 0:
        """ Scalar element """
        raise NotImplementedError()
    
    elif len(V.ufl_element().value_shape()) == 1:
        """ Vector element """
        
        with torch.no_grad():
            assert len(uh.shape) == 3

            uh_np = u_out.vector().get_local()

            pred = model(uh)
            uh_reshape = pred.detach().numpy()
            
            for d in range(uh_reshape.shape[2]):
                uh_np[d::uh_reshape.shape[2]] = uh_reshape[0,:,d]

            u_out.vector().set_local(uh_np)

    else:
        """ Tensor element """
        raise NotImplementedError()

    return u_out


def pred_to_xdmf(model: nn.Module, dataset: FEniCSDataset, output_path: PathLike,
                 overwrite: bool = False) -> None:
    output_path = Path(output_path)
    assert output_path.suffix == ".xdmf", "output_path argument should have suffix .xdmf."

    files_to_overwrite = list(filter(lambda p: p.exists(), [output_path.with_suffix(".xdmf"), output_path.with_suffix(".h5")]))
    if len(files_to_overwrite) > 0:
        if not overwrite:
            print(f"File found at {map(str, files_to_overwrite)}. ")
            if not input("Overwrite? (y/n): ").lower() == "y":
                print("Exiting program.")
                quit()
        for path in files_to_overwrite:
            path.unlink()


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    V = dataset.x_data.function_space

    output_xdmf = df.XDMFFile(str(output_path.with_suffix(".xdmf")))
    output_xdmf.write(V.mesh())

    for k, (x, _) in enumerate(tqdm(dataloader)):
        uh = x
        u_out = pred_checkpoint_torch_to_dolfin(model, uh, V)
        output_xdmf.write_checkpoint(u_out, "uh", k, append=True)

    output_xdmf.close()

    return


if __name__ == "__main__":

    msh = df.UnitSquareMesh(4, 4)
    V = df.VectorFunctionSpace(msh, "CG", 1, 2)
    uh1 = df.Function(V)
    uh2 = df.Function(V)
    print(f"{V.tabulate_dof_coordinates().shape = }")
    print(f"{V.tabulate_dof_coordinates().shape[0] / V.ufl_element().value_size() = }")

    model = nn.Identity()


    uh2 = pred_checkpoint_dolfin_to_dolfin(model, uh1, V, uh2)

    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData("dataset/artificial_learnext", "folders")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))

    pred_to_xdmf(model, dataset, "foo.xdmf")
    Path("foo.xdmf").unlink()
    Path("foo.h5").unlink()
