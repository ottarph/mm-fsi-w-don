import dolfin as df
import numpy as np
import torch

from neuraloperators.deeponet import DeepONet

class Extension:
    pass

class DeepONetExtension(Extension):

    def __init__(self, fspace, deeponet: DeepONet, eval_points: np.ndarray, mask_function: np.ndarray):

        self.fspace = fspace
        self.mesh = fspace.mesh()
        
        self.deeponet = deeponet
        self.mask_torch = torch.tensor(mask_function, dtype=torch.float32)
        self.eval_points_torch = torch.tensor(eval_points, dtype=torch.float32)

        self.harmonic_extension = HarmonicExtension(fspace)

        return
    
    def extend(self, boundary_conditions: df.Function) -> df.Function:

        u_h = self.harmonic_extension.extend(boundary_conditions)

        u_h_np_flat = u_h.vector().get_local()
        u_h_np = np.zeros((self.fspace.dim() // 2, 2))
        u_h_np[:,0] = u_h_np_flat[0::2]
        u_h_np[:,1] = u_h_np_flat[1::2]
        u_h_torch = torch.tensor(u_h_np, dtype=torch.float32)

        u_dno = df.Function(self.fspace)

        with torch.no_grad():
            u_dno_torch = u_h_torch + self.mask_torch * self.deeponet(u_h_torch, self.eval_points_torch)
        u_dno_np_flat = np.zeros_like(u_h_np_flat)
        u_dnp_np = u_dno_torch.detach().numpy()
        u_dno_np_flat[0::2] = u_dnp_np[:,0]
        u_dno_np_flat[1::2] = u_dnp_np[:,1]
        u_dno.vector().set_local(u_dno_np_flat)

        return u_dno
    

class HarmonicExtension(Extension):

    def __init__(self, fspace: df.FunctionSpace):

        self.fspace = fspace
        self.mesh = fspace.mesh()

        trial, test = df.TrialFunction(fspace), df.TestFunction(fspace)
        f = df.Constant((0.0, 0.0))
        self.a = df.inner(df.grad(trial), df.grad(test)) * df.dx
        self.l = df.inner(f, test) * df.dx

        return
    
    def extend(self, boundary_conditions: df.Function) -> df.Function:

        u_h = df.Function(self.fspace)
        bc = df.DirichletBC(self.fspace, boundary_conditions, "on_boundary")

        df.solve(self.a == self.l, u_h, [bc])

        return u_h
    


class BiharmonicExtension(Extension):

    def __init__(self, fspace: df.FunctionSpace):

        self.fspace = fspace
        self.mesh = fspace.mesh()


        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, df.MixedElement(T, T))

        uz = df.TrialFunction(self.FS)
        puz = df.TestFunction(self.FS)
        (u, z) = df.split(uz)
        (psiu, psiz) = df.split(puz)

        dx = df.Measure('dx', domain=self.mesh)

        a = df.inner(df.grad(z), df.grad(psiu)) * dx + df.inner(z, psiz) * dx - df.inner(df.grad(u), df.grad(psiz)) * dx
        L = df.Constant(0.0) * psiu[0] * dx


        self.A = df.assemble(a)

        bc = []
        bc.append(df.DirichletBC(self.FS.sub(0), df.Constant((0.,0.)), 'on_boundary'))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = df.LUSolver(self.A)

        self.L = df.assemble(L)

        return


    def extend(self, boundary_conditions: df.Function) -> df.Function:
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        bc.append(df.DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary'))

        for bci in bc:
            bci.apply(self.L)

        uz = df.Function(self.FS)

        self.solver.solve(uz.vector(), self.L)

        u_, z_ = uz.split(deepcopy=True)

        return df.interpolate(u_, self.fspace)
    
