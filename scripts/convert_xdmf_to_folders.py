import dolfin as df
import numpy as np

from os import PathLike
from pathlib import Path
from tqdm import tqdm
from json import loads


def convert_XDMF_to_folder(xdmf_path: PathLike, folder_path: PathLike,
                           xdmf_file_info: dict, num_checkpoints: int) -> None:
    folder_path = Path(folder_path)

    mesh = df.Mesh()
    xdmf_file = df.XDMFFile(str(xdmf_path))
    xdmf_file.read(mesh)

    if xdmf_file_info["type"] == "vector":
        
        V = df.VectorFunctionSpace(mesh, "CG", xdmf_file_info["degree"], xdmf_file_info["dim"])
        uh = df.Function(V)

        shape_0 = V.tabulate_dof_coordinates().shape[0] // xdmf_file_info["dim"]
        shape_1 = xdmf_file_info["dim"]
        np_shape = [shape_0, shape_1]

        num_digits = int(np.floor(np.log10(num_checkpoints)))+1

        for k in tqdm(range(num_checkpoints)):
            xdmf_file.read_checkpoint(uh, "uh", k)
            uh_loc = uh.vector().get_local()
            uh_np = np.zeros(np_shape, dtype=float)
            for d in range(shape_1):
                uh_np[:,d] = uh_loc[d::xdmf_file_info["dim"]]

            filename = folder_path / f"{k:0{num_digits}d}.npy"
            np.save(filename, uh_np)

        xdmf_file.close()
    

    else:
        xdmf_file.close()
        raise NotImplementedError()


    return


def convert_XDMF_dataset_to_folders(dataset_path: PathLike) -> None:

    dataset_path = Path(dataset_path)

    input_xdmf_path = dataset_path / "input.xdmf"
    output_xdmf_path = dataset_path / "output.xdmf"
    input_folder_path = dataset_path / "input_dir"
    output_folder_path = dataset_path / "output_dir"

    if not input_xdmf_path.exists():
        raise RuntimeError("input xdmf-file does not exist.")
    if not output_xdmf_path.exists():
        raise RuntimeError("input xdmf-file does not exist.")
    
    if input_folder_path.exists():
        if not len(list(input_folder_path.iterdir())) == 0:
            print("Target input data directory is not empty. Continuing might overwrite data.")
            if not input("Continue? (y/n): ").lower() == "y":
                print("Exiting program.")
                quit()
    else:
        input_folder_path.mkdir(parents=True)

    if output_folder_path.exists():
        if not len(list(output_folder_path.iterdir())) == 0:
            print("Target output data directory is not empty. Continuing might overwrite data.")
            if not input("Continue? (y/n): ").lower() == "y":
                print("Exiting program.")
                quit()
    else:
        output_folder_path.mkdir(parents=True)

    with open(dataset_path / "info.json") as infile:
        dataset_info = loads(infile.read())

    print("Converting input file:")
    convert_XDMF_to_folder(input_xdmf_path, input_folder_path, 
                           dataset_info["input"], dataset_info["num_checkpoints"])
    print("Converting output file:")
    convert_XDMF_to_folder(output_xdmf_path, output_folder_path, 
                           dataset_info["output"], dataset_info["num_checkpoints"])


    return


if __name__ == "__main__":
    dataset_path = "dataset/xdmf_dataset"
    convert_XDMF_dataset_to_folders(dataset_path)
