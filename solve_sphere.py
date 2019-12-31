#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

import fenics
import laplace_beltrami_eigensolver as lbe


def make_mesh(num_cells: int) -> fenics.Mesh:
    """Generate mesh for FEM solver.

    """
    mesh = fenics.RectangleMesh(fenics.Point(0, 0),
                                fenics.Point(2 * np.pi, np.pi),
                                num_cells, num_cells)
    pbcs = lbe.periodic_boundary_conditions([0, 0], [2 * np.pi, np.pi],
                                            [True, False])

    return mesh, pbcs


def main():
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], 'NUM-EIGENPAIRS NUM-CELLS DEGREE',
              file=sys.stderr)
        sys.exit(-1)

    num_eigenpairs = int(sys.argv[1])
    num_cells = int(sys.argv[2])
    degree = int(sys.argv[3])

    mesh, pbcs = make_mesh(num_cells)
    tensor_spec = "g[0, 0] = 'pow(sin(x[1]), 2)'; g[1, 1] = 1.0;"
    solver = lbe.LaplaceBeltramiEigensolver(mesh, metric_tensor=tensor_spec,
                                            num_eigenpairs=num_eigenpairs,
                                            degree=degree,
                                            boundary_conditions=pbcs)

    ews, evs = solver.eigenvalues, solver.eigenfunctions
    lbe.plot(ews, evs, mode='color', cmap='RdBu_r', rasterized=True)
    plt.show()


if __name__ == '__main__':
    main()
