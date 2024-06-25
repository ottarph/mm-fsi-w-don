from dolfin import *
import dolfin as df
import numpy as np

from pathlib import Path
import sys, os
from FSIsolver.fsi_solver.solver import SNESProblem

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


class Projector():
    def __init__(self, V):
        self.v = TestFunction(V)
        u = TrialFunction(V)
        form = inner(u, self.v)*dx(V.mesh())
        self.A = assemble(form)
        self.solver = LUSolver(self.A, "mumps")
        self.V = V
    
    def project(self, f):
        L = inner(f, self.v)*dx(self.V.mesh())
        b = assemble(L)
        
        uh = Function(self.V)
        with Timer('LU solver'):
            self.solver.solve(uh.vector(), b)
        
        return uh

class ExtensionOperator(object):
    def __init__(self, mesh, marker, ids):
        self.mesh = mesh
        self.marker = marker
        self.ids = ids

        return
    
    def extend(self, boundary_conditions, params=None):
        """extend the boundary_conditions to the interior of the mesh"""
        raise NotImplementedError

    def custom(self, FSI):
        """custom function for extension operator"""
        return False


class Biharmonic(ExtensionOperator):
    def __init__(self, mesh, marker=None, ids=None, save_extension=False, save_filename=None):
        super().__init__(mesh, marker, ids)

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, MixedElement(T, T))

        uz = TrialFunction(self.FS)
        puz = TestFunction(self.FS)
        (u, z) = split(uz)
        (psiu, psiz) = split(puz)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(z), grad(psiu)) * dx + inner(z, psiz) * dx - inner(grad(u), grad(psiz)) * dx
        L = Constant(0.0) * psiu[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A, "mumps")

        self.L = assemble(L)


    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        uz = Function(self.FS)

        self.solver.solve(uz.vector(), self.L)

        u_, z_ = uz.split(deepcopy=True)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    
class Harmonic(ExtensionOperator):

    def __init__(self, mesh, marker=None, ids=None, save_extension=False, save_filename=None, incremental=False):
        super().__init__(mesh, marker, ids)
        """ Note: the incremental version does not work. Need to add reassembly of all matrices etc. for each time step. """

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)

        self.incremental = incremental
        if self.incremental:
            self.bc_old = Function(self.FS)

        u = TrialFunction(self.FS)
        v = TestFunction(self.FS)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(u), grad(v)) * dx
        L = Constant(0.0) * v[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A, "mumps")

        self.L = assemble(L)


    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        if self.incremental:
            # move mesh with previous deformation
            self.bc_old = boundary_conditions
            up = project(self.bc_old, self.FS)
            upi = Function(self.FS)
            upi.vector().axpy(-1., up.vector())
            ALE.move(self.mesh, up)

        u_ = Function(self.FS)

        self.solver.solve(u_.vector(), self.L)

        if self.incremental:
            # move mesh back
            ALE.move(self.mesh, upi)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    


