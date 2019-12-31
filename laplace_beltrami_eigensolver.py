"""Eigenvalue/eigenfunction solver for the Laplace-Beltrami operator.

This module uses the finite element method to discretize the Laplace-Beltrami
operator of a 2d or 3d manifold on a given chart. Moreover, it is possible to
use adaptive mesh refinement to increase the accuracy in the regions of the
domain where it is needed.

"""


__all__ = ['LaplaceBeltramiEigensolver', 'LaplaceBeltramiEigensolverError',
           'adaptive_eigensolver', 'periodic_boundary_conditions', 'plot']


import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

import dolfin
import fenics
import ufl


DEFAULT_NUM_EIGENPAIRS = 10
DEFAULT_FUNCTION_SPACE = 'Lagrange'
DEFAULT_DEGREE = 2
DEFAULT_TOLERANCE = 1e-8
DEFAULT_MAX_ITERATIONS = 10


class LaplaceBeltramiEigensolverError(Exception):
    """Error during computation of Laplace-Beltrami eigendecomposition.

    """
    pass


class LaplaceBeltramiEigensolver:
    """FEM-based eigensolver for the Laplace-Beltrami operator.

    Attributes
    ----------
    eigenfunctions
        Eigenfunctions of the Laplace-Beltrami operator.
    eigenvalues : np.array
        Eigenvalues of the Laplace-Beltrami operator.

    """
    def __init__(self, mesh: fenics.Mesh, metric_tensor: Optional[str] = None,
                 num_eigenpairs: Optional[int] = DEFAULT_NUM_EIGENPAIRS,
                 function_space: Optional[str] = DEFAULT_FUNCTION_SPACE,
                 degree: Optional[int] = DEFAULT_DEGREE,
                 boundary_conditions: Optional = None) -> None:
        """Compute eigenvalue decomposition of the Laplace-Beltrami operator.

        Parameters
        ----------
        mesh : fenics.Mesh
            Mesh defined over the domain of the Laplace-Beltrami operator
            (this is usually a chart).
        metric_tensor : str, optional
            A specification of the metric tensor (see also the documentation
            for the _make_metric_tensor method below).
        num_eigenpairs : int, optional
            Number of eigenvalue/eigenfunction pairs to (attempt to)
            obtain. The actual number will depend on the convergence of the
            eigensolver.
        function_space : str, optional
            Type of finite element functions.
        degree : int, optional
            Degree for the basis functions.
        boundary_conditions : list, optional
            Constraints on the mesh such as periodic boundary conditions.

        """
        fenics.set_log_level(logging.ERROR)

        self.mesh = mesh
        self.num_eigenpairs = num_eigenpairs
        self.degree = degree
        self.function_space = function_space
        self.boundary_conditions = boundary_conditions
        self.vector_space = fenics.FunctionSpace(mesh, function_space, degree,
                                                 constrained_domain=boundary_conditions)
        self.metric_tensor = self._make_metric_tensor(metric_tensor)
        self.eigensolver = None
        self.eigenvalues, self.eigenfunctions = self.solve()

    def __repr__(self):
        return ('{!r}({!r}, metric_tensor={!r}, num_eigenpairs={!r}, '
                'function_space={!r}, degree={!r}, boundary_conditions={!r})'
                .format(self.__class__.__name__, self.mesh, self.metric_tensor,
                        self.num_eigenpairs, self.function_space, self.degree,
                        self.boundary_conditions))

    def _make_metric_tensor(self, specification: Optional[str]) \
            -> ufl.tensors.ListTensor:
        """Assemble metric tensor.

        Parameters
        ----------
        specification : str, optional
            String containing Python code to define the entries of the metric
            tensor. By default, it is taken to be the identity matrix. The
            code is valid Python code for populating a dictionary `g` indexed
            by tuples.  The entries of the dictionary can be either constants
            or strings of valid UFL code. See also the documentation for
            fenics.functions.expression.Expression.

        Examples
        --------
            The string "g[0, 0] = 'pow(sin(x[1]), 2)'; g[1, 1] = 1.0" gives
            the metric tensor for the unit sphere. Note that the string
            assigned to g[0, 0] is C++ code whereas the assignments
            themselves are valid Python code.

        """
        def zero_matrix(dimension):
            return [[0 for _ in range(dimension)] for _ in range(dimension)]

        def get_dimension(metric_tensor_dict):
            pairs = metric_tensor_dict.keys()
            return max([x for pair in pairs for x in pair]) + 1

        def eval_metric_tensor(spec):
            metric_tensor_dict = {}
            locals_dict = {'g': metric_tensor_dict}
            exec(spec, None, locals_dict)
            return metric_tensor_dict

        vector_space = self.vector_space
        dim = vector_space.ufl_domain().geometric_dimension()

        G = zero_matrix(dim)

        if specification:
            g = eval_metric_tensor(specification)
            dim_g = get_dimension(g)
            if dim != dim_g:
                err_msg = ('The dimension of the metric tensor ({}) does not '
                           'match the dimension of the vector space ({})'
                           .format(dim_g, dim))
                raise LaplaceBeltramiEigensolverError(err_msg)

            degree = vector_space.ufl_element().degree()

            for key, val in g.items():
                i, j = key
                if isinstance(val, str):
                    expr = fenics.Expression(val, degree=degree)
                    expr = fenics.interpolate(expr, vector_space)
                else:
                    expr = val

                G[i][j], G[j][i] = expr, expr
        else:
            for i in range(dim):
                G[i][i] = 1.0

        return ufl.as_tensor(G)

    def _construct_eigenproblem(self, u: ufl.Argument, v: ufl.Argument) \
            -> Tuple[ufl.algebra.Operator, ufl.algebra.Operator]:
        """Construct left- and right-hand sides of eigenvalue problem.

        Parameters
        ----------
        u : ufl.Argument
            A function belonging to the function space under consideration.
        v : ufl.Argument
            A function belonging to the function space under consideration.

        Returns
        -------
        a : ufl.algebra.Operator
            Left hand side form.
        b : ufl.algebra.Operator
            Right hand side form.

        """
        g = self.metric_tensor
        sqrt_g = fenics.sqrt(fenics.det(g))
        inv_g = fenics.inv(g)

        # $a(u, v) = \int_M \nabla u \cdot g^{-1} \nabla v \, \sqrt{\det(g)} \, d x$.
        a = fenics.dot(fenics.grad(u), inv_g * fenics.grad(v)) * sqrt_g
        # $b(u, v) = \int_M u \, v \, \sqrt{\det(g)} \, d x$.
        b = fenics.dot(u, v) * sqrt_g

        return a, b

    def _assemble(self) -> Tuple[fenics.PETScMatrix, fenics.PETScMatrix]:
        """Construct matrices for generalized eigenvalue problem.

        Assemble the left and the right hand side of the eigenfunction
        equation into the corresponding matrices.

        Returns
        -------
        A : fenics.PETScMatrix
            Matrix corresponding to the form $a(u, v)$.
        B : fenics.PETScMatrix
            Matrix corresponding to the form $b(u, v)$.

        """
        V = self.vector_space

        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)

        a, b = self._construct_eigenproblem(u, v)

        A = fenics.PETScMatrix()
        B = fenics.PETScMatrix()

        fenics.assemble(a * fenics.dx, tensor=A)
        fenics.assemble(b * fenics.dx, tensor=B)

        return A, B

    def solve(self) -> Tuple[np.array, List[fenics.Function]]:
        """Solve eigenvalue problem for the Laplace-Beltrami operator.

        Returns
        -------
        eigenvalues : np.array
            Array of eigenvalues.
        eigenfunctions : List[fenics.Function]
            List of eigenfunctions corresponding to the entries in the
            `eigenvalues` array.

        """
        if not fenics.has_linear_algebra_backend("PETSc"):
            raise LaplaceBeltramiEigensolverError('PETSc is unavailable')

        if not fenics.has_slepc():
            raise LaplaceBeltramiEigensolverError('SLEPc is unavailable.')

        A, B = self._assemble()

        self.eigensolver = fenics.SLEPcEigenSolver(A, B)
        parameters = self.eigensolver.parameters
        parameters['problem_type'] = 'pos_gen_non_hermitian'
        parameters['spectrum'] = 'target magnitude'
        parameters['spectral_transform'] = 'shift-and-invert'
        parameters['spectral_shift'] = sys.float_info.epsilon

        self.eigensolver.solve(self.num_eigenpairs)

        n = min(self.num_eigenpairs,
                self.eigensolver.get_number_converged())

        eigenvalues = np.zeros(n, np.float64)
        eigenfunctions = []
        for i in range(n):
            r, c, rx, cx = self.eigensolver.get_eigenpair(i)

            assert np.abs(c) < sys.float_info.epsilon
            eigenvalues[i] = r

            re = fenics.Function(self.vector_space)
            re.vector()[:] = rx

            assert np.linalg.norm(cx, np.inf) < sys.float_info.epsilon
            eigenfunctions.append(re)

        return eigenvalues, eigenfunctions

    def compute_error_estimates(self, eigenvalue: float,
                                eigenfunction: fenics.Function) \
            -> Tuple[np.float, np.float]:
        """Compute error estimates.

        This method computes a posteriori error estimates of two kinds.  The
        first type assumes that the computed eigenvalue is correct and
        verifies that the $L^2$ distance between the right and the left hand
        sides of the variational formulation. The second type of check
        assumes that the eigenfunctions are correct and validates the
        computed eigenvalue against the Rayleigh quotient of the
        eigenfunction.

        Parameters
        ----------
        eigenvalue : float
            An eigenvalue.
        eigenfunction : fenics.Function
            Eigenfunction corresponding to `eigenvalue`.

        Returns
        -------
        l2_dist : float
            $L^2$ distance between the left hand side of the variational
            formulation and the result of multiplying the right hand side by
            `eigenvalue`.
        lambda_dist : float
            Absolute value of the difference between `eigenvalue` and the
            Rayleigh quotient corresponding to `eigenfunction`.

        """
        l, u = eigenvalue, eigenfunction

        a, b = self._construct_eigenproblem(u, u)

        # Compute L2 distance.
        E = fenics.assemble((a - l * b)**2 * fenics.dx(degree=self.degree))
        l2_dist = np.sqrt(np.abs(E))

        # Check Rayleigh quotient.
        deg = self.degree
        rayleigh_quotient = (fenics.assemble(a * fenics.dx(degree=deg))
                             / fenics.assemble(b * fenics.dx(degree=deg)))
        lambda_dist = np.abs(l - rayleigh_quotient)

        return l2_dist, lambda_dist

    def refine_mesh(self, tolerance: float) -> Optional[bool]:
        """Generate refined mesh.

        This method locates cells of the mesh where the error of the
        eigenfunction is above a certain threshold and generates a new mesh
        with finer resolution on the problematic regions.

        For a given eigenfunction $u$ with corresponding eigenvalue
        $\lambda$, it must be true that $a(u, v) = \lambda b(u, v)$ for all
        functions $v$. We make $v$ go through all the basis functions on the
        mesh and locate the cells where the previous identity fails to hold.

        Parameters
        ----------
        tolerance : float
            Criterion to determine whether a cell needs to be refined or not.

        Returns
        -------
        refined_mesh : Optional[fenics.Mesh]
            A new mesh derived from `mesh` with higher level of granularity
            on certain regions. If no refinements are needed, None is
            returned.

        """
        ew, ev = self.eigenvalues[1:], self.eigenfunctions[1:]
        dofs_needing_refinement = set()

        # Find all the degrees of freedom needing refinement.
        for k, (l, u) in enumerate(zip(ew, ev)):
            v = fenics.TrialFunction(self.vector_space)
            a, b = self._construct_eigenproblem(u, v)
            A = fenics.assemble(a * fenics.dx)
            B = fenics.assemble(b * fenics.dx)

            error = np.abs((A - l * B).sum())
            indices = np.flatnonzero(error > tolerance)
            dofs_needing_refinement.update(indices)

        if not dofs_needing_refinement:
            return

        # Refine the cells corresponding to the degrees of freedom needing
        # refinement.
        dofmap = self.vector_space.dofmap()
        cell_markers = fenics.MeshFunction('bool', self.mesh,
                                           self.mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in fenics.cells(self.mesh):
            cell_dofs = set(dofmap.cell_dofs(cell.index()))
            if cell_dofs.intersection(dofs_needing_refinement):
                cell_markers[cell] = True

        return fenics.refine(self.mesh, cell_markers)


def adaptive_eigensolver(mesh: fenics.Mesh,
                         tolerance: float = DEFAULT_TOLERANCE,
                         max_iterations: int = DEFAULT_MAX_ITERATIONS,
                         **kwargs) -> LaplaceBeltramiEigensolver:
    """Run Laplace-Beltrami eigensolver with adaptive mesh refinement.

    Run the Laplace-Belrami eigensolver, refine the mesh based on the a
    posteriori error analysis, and re-run. Stop when the mesh converges or
    the maximum number of iterations is exceeded.

    Parameters
    ----------
    mesh : fenics.Mesh
        Mesh for the domain on which the solution will be computed.
    tolerance : float, optional
        Absolute local error for the eigenpairs.
    max_iterations : int, optional
        Maximum number of iterations.
    kwargs
        Additional keyword arguments to pass to `LaplaceBeltramiEigensolver`.

    Returns
    -------
    laplace_beltrami_eigensolver : LaplaceBeltramiEigensolver
        An instance of LaplaceBeltramiEigensolver containing the eigenvalues,
        eigenfunctions, and all of the auxiliary data structures used during
        the last iteration of the computation.

    """
    kwargs['mesh'] = mesh
    if not mesh:
        raise LaplaceBeltramiEigensolverError('No mesh has been provided.')

    for _ in range(max_iterations):
        solver = LaplaceBeltramiEigensolver(**kwargs)
        mesh = solver.refine_mesh(tolerance)
        if mesh is None:
            break
        else:
            kwargs['mesh'] = mesh

    return solver


def periodic_boundary_conditions(minima: List[float], maxima: List[float],
                                 pbc: List[bool]) -> fenics.SubDomain:
    """Generate periodic boundary conditions on the fly.

    Implement periodic boundary conditions.

    Parameters
    ----------
    minima, maxima : list
        Minimum values and maximum values for each variable.
    pbc : list
        List of booleans specifying whether a coordinate is periodic or not.

    Returns
    -------
    subdomain : fenics.SubDomain
        Compiled code to identify and wrap the periodic boundaries.

    """
    assert len(minima) == len(maxima)
    num_variables = len(minima)

    class PeriodicBoundary(dolfin.SubDomain):
        def __init__(self, tolerance=dolfin.DOLFIN_EPS):
            dolfin.SubDomain.__init__(self, tolerance)
            self.tol = tolerance

        def inside(self, x, on_boundary):
            status = False

            for n in range(num_variables):
                if not pbc[n]:
                    continue

                status |= ((x[n] < minima[n] + self.tol and x[n] > minima[n] - self.tol)
                           and on_boundary)

            return status

        def map(self, x, y):
            for n in range(num_variables):
                y[n] = x[n]
                if pbc[n]:
                    y[n] -= (maxima[n] - minima[n])

    return PeriodicBoundary()


def plot(eigenvalues: np.array, eigenfunctions: List[fenics.Function],
         **kwargs) -> None:
    """Plot eigenvalues and eigenfunctions.

    Convenience function for plotting the results of the eigenfunction
    decomposition.

    """
    num_eigenpairs = len(eigenvalues)
    assert num_eigenpairs == len(eigenfunctions)

    import matplotlib.pyplot as plt

    k = int(np.ceil(np.sqrt(num_eigenpairs - 1)))

    for i, (ew, ev) in enumerate(zip(eigenvalues[1:], eigenfunctions[1:])):
        ax = plt.subplot(k, k, i+1)
        ax.axis('off')
        ax.set_title('$\lambda_{{{}}} = {:.2f}$'.format(i+1, ew))
        fenics.plot(ev, **kwargs)
