import dolfin as df
import numpy as np
import torch


def torch_to_dolfin_scalar(uh: torch.Tensor, V: df.FunctionSpace, 
                    u_out: df.Function | None = None) -> df.Function:
    """
    Assumes uh contains scalars for u in order of scalar-valued dof locations.

    Args:
        uh (torch.Tensor): torch tensor containing scalar-valued function in order of dof-locations.
        V (df.FunctionSpace): Function space to be mapped to.
        u_out (df.Function | None, optional): Supply output vector for inplace. Defaults to None.

    Raises:
        ValueError: Requires `V` to be scalar-valued function space.

    Returns:
        df.Function: `uh` converted to dolfin.
    """

    if not len(V.ufl_element().value_shape()) == 0:
        raise ValueError
    
    assert uh.device == torch.device("cpu")

    if u_out is None:
        u_out = df.Function(V)

    uh_reshape = uh.detach().numpy().flatten()
    assert uh_reshape.shape[-1] == uh.shape[-1]

    u_out.vector()[:] = uh_reshape

    return u_out


def torch_to_dolfin_vector(uh: torch.Tensor, V: df.FunctionSpace, 
                    u_out: df.Function | None = None, scratch: np.ndarray | None = None) -> df.Function:
    """
    Assumes uh contains vectors for u in order of scalar-valued dof locations.

    Args:
        uh (torch.Tensor): torch tensor containing vector-valued function in order of dof-locations.
        V (df.FunctionSpace): Function space to be mapped to.
        u_out (df.Function | None, optional): Supply output vector for inplace. Defaults to None.
        scratch (np.ndarray | None, optional): Supply pre-allocated workspace.

    Raises:
        NotImplementedError: Requires `uh` to be of three-dim tensor.

    Returns:
        df.Function: `uh` converted to dolfin.
    """
    if not len(uh.shape) == 3:
        raise NotImplementedError
    if not len(V.ufl_element().value_shape()) == 1:
        raise ValueError
    
    assert uh.device == torch.device("cpu")

    if u_out is None:
        u_out = df.Function(V)
    if scratch is None:
        scratch = u_out.vector().get_local()
    else:
        assert scratch.shape == (uh.shape[1] * uh.shape[2],)

    uh_reshape = uh.detach().numpy()

    for d in range(uh_reshape.shape[2]):
        scratch[d::uh_reshape.shape[2]] = uh_reshape[0,:,d]
    
    u_out.vector().set_local(scratch)

    return u_out


def torch_to_dolfin(uh: torch.Tensor, V: df.FunctionSpace, 
                    u_out: df.Function | None = None, scratch: np.ndarray | None = None) -> df.Function:
    """
    Assumes uh contains scalars or vectors for u in order of scalar-valued dof locations.

    Args:
        uh (torch.Tensor): torch tensor containing scalar- or vector-valued function in order of dof-locations.
        V (df.FunctionSpace): Function space to be mapped to.
        u_out (df.Function | None, optional): Supply output vector for inplace. Defaults to None.
        scratch (np.ndarray | None, optional): Supply pre-allocated workspace.

    Raises:
        NotImplementedError: For now only implemented for scalar- and vector-valued functions.
        NotImplementedError: Only works on dolfin vector functions for now.

    Returns:
        df.Function: `uh` converted to dolfin.
    """

    if u_out is None:
        u_out = df.Function(V)

    if len(V.ufl_element().value_shape()) == 0:
        torch_to_dolfin_scalar(uh, V, u_out)
    elif len(V.ufl_element().value_shape()) == 1:
        torch_to_dolfin_vector(uh, V, u_out, scratch)
    else:
        raise NotImplementedError

    return u_out
