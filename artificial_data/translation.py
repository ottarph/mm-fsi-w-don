import dolfin as df
import numpy as np

def translate_entity_f(parent_mesh, source_f, child_mesh, tags):
    '''Create entity function on child mesh'''
    child2parent = parent_mesh.data().array('parent_vertex_indices', 0)
    vertex_map = {v: k for k, v in enumerate(child2parent)}

    edim = source_f.dim()

    _, e2v = child_mesh.init(edim, 0), child_mesh.topology()(edim, 0)

    child_f = df.MeshFunction('size_t', child_mesh, edim, 0)
    child_entities = {tuple(sorted(e.entities(0))): e.index() for e in df.SubsetIterator(child_f, 0)}
    for tag in tags:
        for parent_entity in df.SubsetIterator(source_f, tag):
            parent_vertices = parent_entity.entities(0)
            # Now encode them in child
            child_vertices = tuple(sorted([vertex_map[pv] for pv in parent_vertices]))
            # Look for the matching child entity
            child_entity = child_entities[child_vertices]
            child_f[child_entity] = tag
    return child_f


def translate_function(from_u, to_facet_f, from_facet_f, shared_tags, to_u=None, tol_=1E-12):
    '''
    If tu_u and from_u are 2 functions on different domains that share 
    facets we transfer boundary data to to_u from from_u.
    '''
    if to_u is None:
        to_u = df.Function(df.FunctionSpace(to_facet_f.mesh(), from_u.ufl_element()))

    shape = to_u.ufl_shape    
    assert shape == from_u.ufl_shape
    assert len(shape) == 1  # We will only use this for vector spaces
    
    assert shared_tags
    assert to_facet_f.dim() == from_facet_f.dim()

    Vto = to_u.function_space()
    assert Vto.mesh().id() == to_facet_f.mesh().id()
    assert Vto.mesh().topology().dim() - 1 == to_facet_f.dim()

    Vfrom = from_u.function_space()
    assert Vfrom.mesh().id() == from_facet_f.mesh().id()
    assert Vfrom.mesh().topology().dim() - 1 == from_facet_f.dim()

    assert Vfrom.ufl_element() == Vto.ufl_element()

    to_dofs, from_values = [], []
    for dim in range(shape[0]):
        for tag in shared_tags:
            from_u_dim = df.Function(Vfrom.sub(dim).collapse())
            df.assign(from_u_dim, from_u.sub(dim))
            bc_from = df.DirichletBC(Vfrom.sub(dim), from_u_dim, from_facet_f, tag)
            
            this_from_dofs = list(bc_from.get_boundary_values().keys())
            from_values.extend(bc_from.get_boundary_values().values())
            
            # Bc dofs need to find the right permutation ...
            bc_to = df.DirichletBC(Vto.sub(dim), df.Constant(0), to_facet_f, tag)
            this_to_dofs = list(bc_to.get_boundary_values().keys())

            # ... based on position of dofs
            x = Vfrom.tabulate_dof_coordinates()
            y = Vto.tabulate_dof_coordinates()[this_to_dofs]

            for from_dof in this_from_dofs:
                dist = np.linalg.norm(y - x[from_dof], 2, axis=1)
                assert len(dist) == len(y)
                match_index = np.argmin(dist)
                assert dist[match_index] < tol_, (dist[match_index], dim, tag)

                to_dofs.append(this_to_dofs[match_index])

    to_values = to_u.vector().get_local()
    to_values[to_dofs] = from_values

    to_u.vector().set_local(to_values)

    return to_u

