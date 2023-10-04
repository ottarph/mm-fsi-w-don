import dolfin as df
import numpy as np
import argparse
import json

from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=Path("artificial_data/data"), type=Path)
    parser.add_argument("--dataset-dir", default=Path("dataset/artificial_learnext"), type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    work_dir: Path = args.work_dir
    dataset_dir: Path = args.dataset_dir
    overwrite: bool = args.overwrite


    from custom_loads import loads

    files_to_be_used = [work_dir / "fluid.h5", work_dir / "solid.h5"]
    files_to_be_used.extend([work_dir / f"displacements{l}.xdmf" for l in range(1, len(loads)+1)])
    files_to_be_used.extend([work_dir / f"displacements{l}.h5" for l in range(1, len(loads)+1)])
    
    created_files = [dataset_dir / "input.xdmf", dataset_dir / "output.xdmf",
                     dataset_dir / "input.h5", dataset_dir / "output.h5",
                     dataset_dir / "info.json"]

    files_to_overwrite = list(filter(lambda p: p.exists(), files_to_be_used + created_files))


    if len(files_to_overwrite) > 0:
        if not overwrite:
            if input(f"Overwrite files:\n\t{list(map(str, files_to_overwrite))}?\n\t(y/n): ").lower() != "y":
                print("\nQuitting.")
                quit()

        for file in files_to_overwrite:
            file.unlink()

    if not work_dir.exists():
        work_dir.mkdir(parents=True)
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    df.set_log_active(False)

    """ Create fluid and solid mesh. """

    print("Creating fluid and solid mesh.")

    fluid_mesh_path = work_dir / "fluid.h5"
    solid_mesh_path = work_dir / "solid.h5"

    
    from make_mesh import create_mesh, translate_entity_f
    mesh, entity_fs, mapping, params, _ = create_mesh(verbosity=0)
    # Submeshes
    cell_f = entity_fs[mesh.topology().dim()]

    fluid_mesh = df.SubMesh(mesh, cell_f, params['fluid'])
    solid_mesh = df.SubMesh(mesh, cell_f, params['solid'])

    facet_f = entity_fs[mesh.topology().dim()-1]
    
    fluid_boundaries = translate_entity_f(fluid_mesh, facet_f, fluid_mesh, mapping[params['fluid']])
    solid_boundaries = translate_entity_f(solid_mesh, facet_f, solid_mesh, mapping[params['solid']])

    with df.HDF5File(mesh.mpi_comm(), str(fluid_mesh_path), 'w') as h5:
        h5.write(fluid_mesh, 'mesh')
        h5.write(fluid_boundaries, 'boundaries')

    with df.HDF5File(mesh.mpi_comm(), str(solid_mesh_path), 'w') as h5:
        h5.write(solid_mesh, 'mesh')
        h5.write(solid_boundaries, 'boundaries')


    """ Create solid displacements from custom defined surface loads """

    print("\n\nCreating solid displacements from custom surface loads")

    from make_displacements import make_displacements
    
    # ----6----
    # 4       9
    # ----6----

    displacement_bcs = {4: df.Constant((0.0, 0.0))}
    volume_load = df.Constant((0.0, 0.0))

    assert len(loads) < 10, "Numbering of displacement files will cause error."

    for l, load in enumerate(tqdm(loads), start=1):
        make_displacements(solid_mesh_path, load, work_dir / f"displacements{l}.xdmf")


    """ Compute deformation fields over fluid mesh. """

    print("\n\nComputing mesh deformation fields over fluid mesh.")

    from make_dataset import convert_dataset


    fluid_tags = set(fluid_boundaries.array()) - set((0, ))
    iface_tags = {6, 9}
    zero_displacement_tags = fluid_tags - iface_tags

    domain_info = {
        "solid_boundaries": solid_boundaries,
        "fluid_boundaries": fluid_boundaries,
        "iface_tags": iface_tags,
        "zero_displacement_tags": zero_displacement_tags
    }

    harmpath = Path(dataset_dir / "input.xdmf")
    biharmpath = Path(dataset_dir / "output.xdmf")


    harmfile = df.XDMFFile(str(harmpath))
    biharmfile = df.XDMFFile(str(biharmpath))


    harmfile.write(fluid_mesh)
    biharmfile.write(fluid_mesh)

    for n in tqdm(range(1, len(loads)+1)):

        filepath = work_dir / f"displacements{n}.xdmf"

        order = 1
        convert_dataset(filepath, harmfile, biharmfile, fluid_mesh, solid_mesh, domain_info, checkpoint_offset=(n-1)*101, order=order)

    harmfile.close()
    biharmfile.close()

    info_dict = {
        "input": {
            "type": "vector",
            "dim": 2,
            "degree": 1,
            "label": "uh"
        },
        "output": {
            "type": "vector",
            "dim": 2,
            "degree": 1,
            "label": "uh"
        },
        "num_checkpoints": 101 * len(loads)
    }
    print(info_dict)
    with open(dataset_dir / "info.json", "w") as outfile:
        json.dump(info_dict, outfile, indent=4)


    """ Convert created XDMF-dataset to folders style for faster access. """
    
    print("\n\nConverting XDMF-dataset to folders style for faster access.")

    from scripts.convert_to_folders import convert_XDMF_dataset_to_folders

    convert_XDMF_dataset_to_folders(dataset_dir)



    