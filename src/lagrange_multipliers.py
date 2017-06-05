"""
Use Lagrange multipliers to preform A Priori Lexicographical Multi-Objective
Optimization.

See math/lagrange_multipliers.pdf for a full explanation of the algorithm.

Written by Zachary Ferguson and Alec Jacobson.
"""

import pdb

import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import includes

from spqr import qr as spqr


def LagrangeMultiplierMethod(H, f):
    """
    LEXMIN Solve the multi-objective minimization problem:

        min  {E1(x), E2(x), ... , Ek(x)}
         x

    where

        Ei = 0.5 * x.T * H[i] * x + x.T * f[i]

    and Ei is deemed "more important" than Ei+1 (lexicographical ordering):
    https://en.wikipedia.org/wiki/Multi-objective_optimization#A_priori_methods

    Inputs:
        H - k-cell list of n by n sparse matrices, so that H[i] contains the
            quadratic coefficients of the ith energy.
        f - k-cell list of n by 1 vectors, so that f[i] contains the linear
            coefficients of the ith energy
    Outputs:
        z - n by 1 solution vector
    Note:
        This method is bad for multiple reasons:
            1. qr factorization on the ith set of constraints will be very
               expensive,
            2. this is due to the fact that the number of non-zeros will be
               O(n2^i), and
    """
    is_sparse = scipy.sparse.issparse(H[0])

    k = len(H) # Number of energies
    n = H[0].shape[0]

    # C = H{1};
    C = H[0]
    # D = -f{1};
    d = -f[0]

    # for i = 1:k
    for i in range(k):
        # [Q,R,E] = qr(C');
        if(is_sparse):
            Q, R, E, rank = spqr(C.T)
            Q = Q.tocsc()
            R = R.tocsc()
        else:
            Q, R, E = scipy.linalg.qr(C.T, pivoting=True)

        # nc = find(any(R,2),1,'last');
        nonzero_rows = R.nonzero()[0]
        if(nonzero_rows.shape[0] > 0):
            nc = nonzero_rows[-1] + 1
        else:
            nc = 0

        # if nc == size(C,1) || i==k
        if nc == C.shape[0]:
            # Z = C \ D; Z = Z(1:n);
            return (scipy.sparse.linalg.spsolve(C.tocsc(), d)[:n] if is_sparse
                else numpy.linalg.solve(C, d)[:n])

        # col(Q) = col(C.T) = row(C)
        m = C.shape[0]
        C = (Q[:, :nc].dot(R[:nc, :nc])).T
        d = d[E[:nc]]
        if(i == (k - 1)):
            return (scipy.sparse.linalg.spsolve(C.tocsc(), d)[:n] if is_sparse
                else numpy.linalg.solve(C, d)[:n])

        # A = sparse(size(C,1),size(C,1));
        # A(1:n,1:n) = H{i+1};
        # C = [A C';C sparse(size(C,2),size(C,2))];
        if(is_sparse):
            A = H[i + 1].tocoo()
            A = scipy.sparse.coo_matrix((A.data, (A.row, A.col)), shape=(m, m))
            C = scipy.sparse.vstack([scipy.sparse.hstack([A, C.T]),
                scipy.sparse.hstack([C,
                    scipy.sparse.csc_matrix((C.shape[0], C.shape[0]))])])
        else:
            A = numpy.zeros((m, m))
            A[:H[i + 1].shape[0], :H[i + 1].shape[1]] = H[i + 1]
            C = numpy.vstack([numpy.hstack([A, C.T]),
                numpy.hstack([C, numpy.zeros((C.shape[0], C.shape[0]))])])

        # B = zeros(size(C,1),1);
        b = numpy.zeros((m, d.shape[1]))
        # B(1:n) = -f{i+1};
        b[:f[i + 1].shape[0]] = -f[i + 1]
        # D = [B;D];
        d = numpy.vstack([b, d])

if __name__ == "__main__":
    import time_aplmoo_method
    time_aplmoo_method.time_aplmoo_method(LagrangeMultiplierMethod,
        print_energy=True)
