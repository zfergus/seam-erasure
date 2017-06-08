"""
Utility functions for building inequality matrices.
Author: Zachary Ferguson
"""

import numpy
import scipy.sparse
__cvxopt_found = True
try:
    import cvxopt
except ImportError:
    __cvxopt_found = False


def number_bounds_to_array(shape, bounds):
    """
    Converts number bounds to full array of shape.
    If type(bounds[i]) == number then
        bounds[i] = numpy.full((#A, #b), bounds[i])
    """
    array_bounds = list(bounds[:])
    for i, bound in enumerate(bounds):
        if(isinstance(bound, (int, long, float, complex))):
            array_bounds[i] = numpy.full(shape, float(bound))
    return array_bounds


def BoundingMatrix(shape, bounds):
    """
    Constructs an inequality matrix for bounding x in the bounds.
        Gx <= h
    Inputs:
        shape  - shape of the x vector, (N, M)
        bounds - A length two tuple where bounds[0] <= x <= bounds[1].
    Outputs:
        G - 2NxN matrix such that Gx <= h
        h - 2NxM vector where the first N are bound[0] and the second N are
            bounds[1]
    """
    n, m = shape

    # First diagonal is -I so that -x < bounds[0]
    # Second diagonal is I so that x < bounds[1]
    data = numpy.hstack([-numpy.ones(n), numpy.ones(n)])
    ijs = (numpy.arange(2 * n),
        numpy.hstack([numpy.arange(n), numpy.arange(n)]))
    G = scipy.sparse.coo_matrix((data, ijs), shape=(2 * n, n))

    # First N are bounds[0], second N are bounds[1]
    h = numpy.vstack(number_bounds_to_array(shape, bounds))

    return G, h


if(__cvxopt_found):
    # Construct an inequality constraint matrix, Gx <= h, for
    # a <= x <= b
    def CVXOPTBoundingMatrix(shape, bounds):
        """
        Constructs an inequality matrix for bounding x in the bounds.
            Gx <= h
        Returns G and h as CVXOPT spmatrix and matrix.
        See: BoundingMatrix
        """
        G, h = BoundingMatrix(shape, bounds)
        G = cvxopt.spmatrix(G.data, G.row.astype(int), G.col.astype(int))
        h = cvxopt.matrix(h)
        return G, h

    def cvxopt_solve_all_depth(shape, P, q, G=None, h=None, A=None, b=None,
            solver = None):
        """
        Wrapper around CVXOPT's QP solver that solve for all channels.
        Input:
            shape - the shape of the linear term, q.
            !(shape) - same as CVXOPT QP
        Output:
            Numpy array of the solved channels.
        """
        # sol = cvxopt.solvers.qp(P, q, G=G, h=h, A=A, b=b, solver=solver)
        x = numpy.empty(shape)
        for i in range(shape[1]):
            hi = None if h is None else h[:, i]
            bi = None if b is None else b[:, i]
            result  = cvxopt.solvers.qp(P, q[:, i], G=G, h=hi, A=A, b=bi,
                solver=solver)
            # result  = cvxopt.solvers.qp(P, q[:, i], solver = solver)
            x[:, i] = numpy.array(result["x"]).flatten()
        return x

if __name__ == "__main__":
    n = 100
    G, h = BoundingMatrix((n, 1), (0, 1))
    lessthan = True
    for i in range(1000):
        x = numpy.random.rand(n, 1)
        lessthan &= (G.dot(x) <= h).all()
        if(not lessthan):
            break
    print("Gx <= h: %s" % lessthan)
