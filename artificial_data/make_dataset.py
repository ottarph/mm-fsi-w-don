import dolfin as df
import numpy as np

from pathlib import Path
from tqdm import tqdm
from os import PathLike
from translation import translate_function


def convert_dataset(path: PathLike, harm_file: df.XDMFFile, biharm_file: df.XDMFFile,
                    fluid_mesh: df.Mesh, solid_mesh: df.Mesh, domain_info: dict, 
                    checkpoint_offset: int = 0, order: int = 1) -> None:
    
    solid_boundaries = domain_info["solid_boundaries"]
    fluid_boundaries = domain_info["fluid_boundaries"]
    iface_tags = domain_info["iface_tags"]
    zero_displacement_tags = domain_info["zero_displacement_tags"]

    CG2_solid = df.VectorFunctionSpace(solid_mesh, "CG", 2)
    CG2_fluid = df.VectorFunctionSpace(fluid_mesh, "CG", 2)

    u_solid_cg2 = df.Function(CG2_solid)

    V_fluid = df.VectorFunctionSpace(fluid_mesh, "CG", order)

    """ Make Laplace solver """
    u_harm_V = df.Function(V_fluid)
    u_harm_cg2 = df.Function(CG2_fluid)

    u_cg2 = df.TrialFunction(CG2_fluid)
    v_cg2 = df.TestFunction(CG2_fluid)

    a_harm = df.inner( df.grad(u_cg2), df.grad(v_cg2) ) * df.dx
    l_harm = df.inner( df.Constant((0.0, 0.0)), v_cg2) * df.dx

    A_harm = df.as_backend_type(df.assemble(a_harm))
    L_harm = df.assemble(l_harm)
    
    solver_harm = df.PETScLUSolver(A_harm)


    """ Make biharmonic solver """
    u_biharm_V = df.Function(V_fluid)
    u_biharm_cg2 = df.Function(CG2_fluid)

    # Mixed formulation
    T = df.VectorElement("CG", fluid_mesh.ufl_cell(), 2)
    FS = df.FunctionSpace(fluid_mesh, df.MixedElement(T, T))
    uz = df.TrialFunction(FS)
    puz = df.TestFunction(FS)
    u, z = df.split(uz)
    psiu, psiz = df.split(puz)

    uz_h = df.Function(FS)

    a_biharm = df.inner( df.grad(z), df.grad(psiu) ) * df.dx + \
                df.inner(z, psiz) * df.dx + \
                -df.inner( df.grad(u), df.grad(psiz) ) * df.dx
    l_biharm = df.inner( df.Constant((0.0, 0.0)), psiu) * df.dx

    A_biharm = df.as_backend_type(df.assemble(a_biharm))
    L_biharm = df.assemble(l_biharm)

    solver_biharm = df.PETScLUSolver(A_biharm)

    with df.XDMFFile(str(path)) as infile:

        for k in tqdm(range(101), leave=False):
            infile.read_checkpoint(u_solid_cg2, "uh", k)

            uh_fluid = translate_function(from_u=u_solid_cg2,
                                  from_facet_f=solid_boundaries,
                                  to_facet_f=fluid_boundaries,
                                  shared_tags=iface_tags)
            
            bcs = [df.DirichletBC(CG2_fluid, uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
            null = df.Constant((0.0, 0.0))
            bcs.extend([df.DirichletBC(CG2_fluid, null, fluid_boundaries, tag) for tag in zero_displacement_tags])

            for bc in bcs:
                bc.apply(A_harm)
                bc.apply(L_harm)

            solver_harm.solve(u_harm_cg2.vector(), L_harm)
            u_harm_V = df.interpolate(u_harm_cg2, V_fluid)
            harm_file.write_checkpoint(u_harm_V, "uh", checkpoint_offset + k, append=True)

            bcs = [df.DirichletBC(FS.sub(0), uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
            null = df.Constant((0.0, 0.0))
            bcs.extend([df.DirichletBC(FS.sub(0), null, fluid_boundaries, tag) for tag in zero_displacement_tags])

            for bc in bcs:
                bc.apply(A_biharm)
                bc.apply(L_biharm)

            solver_biharm.solve(uz_h.vector(), L_biharm)
            u_biharm_cg2, zh = uz_h.split(deepcopy=True)
            u_biharm_V = df.interpolate(u_biharm_cg2, V_fluid)
            biharm_file.write_checkpoint(u_biharm_V, "uh", checkpoint_offset + k, append=True)


    return


if __name__ == "__main__":


    working_dir = "artificial_data/data"

    solid_mesh = df.Mesh()
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+'/solid.h5', 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+'/solid.h5', 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')

    fluid_mesh = df.Mesh()
    with df.HDF5File(fluid_mesh.mpi_comm(), working_dir+'/fluid.h5', 'r') as h5:
        h5.read(fluid_mesh, 'mesh', False)

    tdim = fluid_mesh.topology().dim()
    fluid_boundaries = df.MeshFunction('size_t', fluid_mesh, tdim-1, 0)
    with df.HDF5File(fluid_mesh.mpi_comm(), working_dir+'/fluid.h5', 'r') as h5:
        h5.read(fluid_boundaries, 'boundaries')

    fluid_tags = set(fluid_boundaries.array()) - set((0, ))
    iface_tags = {6, 9}
    zero_displacement_tags = fluid_tags - iface_tags

    domain_info = {
        "solid_boundaries": solid_boundaries,
        "fluid_boundaries": fluid_boundaries,
        "iface_tags": iface_tags,
        "zero_displacement_tags": zero_displacement_tags
    }


    harmpath = Path("artificial_data/input.xdmf")
    biharmpath = Path("artificial_data/output.xdmf")

    if harmpath.exists():
        if input(f"overwrite {harmpath}? (y/n): ") == "y":
            harmpath.unlink()
        else:
            print("File already exists, not overwriting.")
            quit()
    
    if biharmpath.exists():
        if input(f"overwrite {biharmpath}? (y/n): ") == "y":
            biharmpath.unlink()
        else:
            print("File already exists, not overwriting.")
            quit()

    harmfile = df.XDMFFile(str(harmpath))
    biharmfile = df.XDMFFile(str(biharmpath))


    harmfile.write(fluid_mesh)
    biharmfile.write(fluid_mesh)

    for n in tqdm(range(1, 6+1)):

        filepath = f"artificial_data/data/displacements{n}.xdmf"
        order = 1
        convert_dataset(filepath, harmfile, biharmfile, fluid_mesh, solid_mesh, domain_info, checkpoint_offset=(n-1)*101, order=order)

    harmfile.close()
    biharmfile.close()


