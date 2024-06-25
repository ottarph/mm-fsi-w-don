import dolfin as df
import torch
import numpy as np

from os import PathLike
from pathlib import Path

from deeponet_extension.deeponet import DeepONet
from deeponet_extension.loading import load_deeponet_problem

from FSIsolver.extension_operator.extension import ExtensionOperator
from deeponet_extension.mask import poisson_mask_custom


class DeepONetExtension(ExtensionOperator):

    def __init__(self, mesh: df.Mesh, deeponet: DeepONet | PathLike, T_switch: float = 0.0, mask_rhs: str | None = None, torch_device: str = "cuda", silent: bool = False):

        super().__init__(mesh, marker=None, ids=None)
        
        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.F = df.FunctionSpace(self.mesh, T)
        self.iter = -1

        # harmonic extension 
        uh = df.TrialFunction(self.F)
        v = df.TestFunction(self.F)

        a = df.inner(df.grad(uh), df.grad(v))*df.dx
        L = df.Constant(0.0) * v[0] *df.dx
        A = df.assemble(a)
        
        bc = df.DirichletBC(self.F, df.Constant((0.,0.)), 'on_boundary')
        bc.apply(A)

        self.solver_harmonic = df.LUSolver(A, "mumps")
        self.rhs_harmonic = df.assemble(L)

        self.uh = df.Function(self.F)
        self.u_ = df.Function(self.F)
        self.uh_np_flat = np.zeros_like(self.uh.vector()[:])
        self.uh_np = np.zeros((self.F.dim() // 2, 2))
        self.u_np_flat = np.zeros_like(self.uh_np_flat)


        # DeepONet model
        if not isinstance(deeponet, DeepONet):
            deeponet = Path(deeponet)
            deeponet, ld_mash_rhs = load_deeponet_problem(deeponet / "problem.json", self.F, deeponet)
            if mask_rhs is None:
                mask_rhs = ld_mash_rhs

        self.torch_device = torch.device(torch_device)

        self.deeponet = deeponet
        self.deeponet.to(self.torch_device)
        self.deeponet.eval()


        # mask for adjusting deeponet correction
        if mask_rhs is None:
            # Masking function custom made for specific domain.
            mask_rhs = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"

        V_scal = df.FunctionSpace(self.mesh, "CG", self.F.ufl_element().degree())
        poisson_mask = poisson_mask_custom(V_scal, mask_rhs, normalize=True)

        self.mask_np = poisson_mask.vector().get_local().reshape(-1,1)
        # Mask needs to have shape (num_vertices, 1) to broadcast correctly in
        # multiplication with correction of shape (num_vertices, 2).

        self.mask_tensor = torch.tensor(self.mask_np, dtype=torch.float32, device=self.torch_device)

        # Construct eval_points
        eval_points = np.copy(self.F.tabulate_dof_coordinates()[::2])
        self.eval_points_torch = torch.tensor(eval_points, dtype=torch.float32, device=self.torch_device)

        # # Time to switch from harmonic to deeponet extension
        self.T_switch = T_switch

        self.silent = silent

        return

    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params):
        """ Torch-corrected extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]

        # harmonic extension
        bc = df.DirichletBC(self.F, boundary_conditions, 'on_boundary')
        bc.apply(self.rhs_harmonic)
        self.solver_harmonic.solve(self.uh.vector(), self.rhs_harmonic)

        with df.Timer("deeponet"):
            if t < self.T_switch:
                self.u_.vector()[:] = self.uh.vector()[:]
            
            else:
                if not self.silent:
                    print("deeponet extension")

                self.uh_np_flat[:] = self.uh.vector()[:]
                self.uh_np[:,0] = self.uh_np_flat[0::2]
                self.uh_np[:,1] = self.uh_np_flat[1::2]

                self.uh_torch = torch.tensor(self.uh_np, dtype=torch.float32, device=self.torch_device)

                with torch.no_grad():
                    u_torch = self.uh_torch + self.mask_tensor * self.deeponet(self.uh_torch, self.eval_points_torch)

                u_np = u_torch.detach().cpu().numpy()
                self.u_np_flat[0::2] = u_np[:,0]
                self.u_np_flat[1::2] = u_np[:,1]

                self.u_.vector()[:] = self.u_np_flat[:]

        return self.u_

