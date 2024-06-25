import dolfin as df
import numpy as np


def poisson_mask_custom(V: df.FunctionSpace, f_str: str, normalize: bool = False) -> df.Function:
    """
        -Delta u = f in Omega
               u = 0 on dOmega
    """

    def boundary(x, on_boundary):
        return on_boundary
    u0 = df.Constant(0.0)
    
    bc = df.DirichletBC(V, u0, boundary)

    f = df.Expression(f_str, degree=5)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    l = f * v * df.dx

    uh = df.Function(V)
    df.solve(a == l, uh, bc)
    
    if normalize:
        uh.vector()[:] /= np.max(uh.vector()[:]) # Normalize mask to have sup-norm 1.

    return uh
