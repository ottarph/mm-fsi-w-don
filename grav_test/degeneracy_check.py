import dolfin as df
import numpy as np

def get_degenerate_cells(u: df.Function, tensor_dg_order: int = 6) -> df.Function:

    msh = u.function_space().mesh()

    F = df.Identity(len(u)) + df.grad(u)

    V = df.TensorFunctionSpace(msh, 'DG', tensor_dg_order)
    ndofs_per_dim = V.sub(0).collapse().dolfin_element().space_dimension()

    v = df.TrialFunction(V)
    dv = df.TestFunction(V)

    a = df.inner(v, dv)*df.dx
    L = df.inner(F, dv)*df.dx

    cell_stats = []
    for cell in df.cells(msh):
        A = df.assemble_local(a, cell=cell)
        b = df.assemble_local(L, cell=cell)
        T = np.linalg.solve(A, b)

        # Sample in dofs
        dets = []
        T_at_dofs = T.reshape((4, ndofs_per_dim)).T
        for T in T_at_dofs:
            T = T.reshape((2, 2))
            dets.append(np.linalg.det(T))
        cell_stats.append((min(dets), max(dets)))
    cell_stats = np.array(cell_stats)

    bad_cells = np.where(cell_stats[:, 0] < 0, -np.ones(cell_stats.shape[0], dtype=np.float64), np.ones(cell_stats.shape[0], dtype=np.float64))

    dg_func = df.Function(df.FunctionSpace(msh, "DG", 0))
    dg_func.vector()[:] = bad_cells

    return dg_func

