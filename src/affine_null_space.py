# -*- coding: utf-8 -*-
"""
Given a system Ax = b, determine a matrix N spanning the right null space of A
and a feasible solution x0 so that:
    A * (N * y + x0) = b  for any y

Written by Zachary Ferguson
"""

import numpy
import scipy.sparse
# import cvxopt

import pdb

import includes

from luq import luq
from spqr import qr as spqr, permutation_from_E

# from inequalities import BoundingMatrix, cvxopt_solve_all_depth

import test_util
from bcolors import bcolors

import warnings
warnings.simplefilter("ignore", scipy.sparse.SparseEfficiencyWarning)


def dense_affine_null_space(A, b, tolerance = 1e-8, method = "qr"):
    """
    Version of affine_null_space for dense matrices.
    Inputs:
        method - either
            "qr"  for QR decomposition
            "svd" for SVD decomposition
    (See affine_null_space for full list of Inputs/Outputs)
    """

    # print("dense_affine_null_space: ", A.shape, b.shape, tolerance, method)

    if(scipy.sparse.issparse(A)):
        A = A.toarray()
    if(scipy.sparse.issparse(b)):
        b = b.toarray()

    method = method.lower()
    if(method == "qr"):
        Q, R, E = scipy.linalg.qr(A.T, pivoting=True)
        P = permutation_from_E(E).toarray()

        # Rank of A.T = Row rank of A
        # nc = find(any(abs(R)>tol,2),1,'last');
        nonzero_rows = (abs(R) > tolerance).nonzero()[0]
        if(nonzero_rows.shape[0] > 0):
            nc = nonzero_rows[-1] + 1
        else:
            nc = 0

        # Q  = [Q1, Q2]
        # Q1 = Q(:,1:nc)
        Q1 = Q[:, :nc]
        # Q2 = Q(:,nc+1:end)
        # N  = Q2
        N = Q2 = Q[:, nc:]
        # R = [R1, R2; 0]
        R1 = R[:nc, :nc]

        # A possibly non-unique solution
        # x0 = Q1*(R(1:nc,1:nc)'\(E(:,1:nc)'*(b)))
        b1 = P.T.dot(b)[:nc]
        y1 = numpy.linalg.solve(R1.T, b1)
        x0 = Q1.dot(y1)

    elif(method == "svd"):
        U, s, VT = numpy.linalg.svd(A)
        V = VT.T
        singular_i = (s < 1e-15).nonzero()[0]
        N = V[:, singular_i]
        pseudo_invert = numpy.vectorize(lambda x: (1 / float(x))
            if abs(x) > 1e-8 else 0.0)
        S_plus = numpy.diag(pseudo_invert(s))
        x0 = V.dot(S_plus).dot(U.T).dot(b)

    else:
        raise ValueError("Invalild method for solving for the affine null \
            space, %s." % method)

    return N, x0


def sparse_affine_null_space(A, b, tolerance = 1e-8, method = "qr"):
    """
    Version of affine_null_space for sparse matrices.
    Inputs:
        method - either
            "qr"  for QR decomposition
            "luq" for LUQ decomposition
    (See affine_null_space for full list of Inputs/Outputs)
    """

    # print("sparse_affine_null_space: ", A.shape, b.shape, tolerance, method)

    if(not scipy.sparse.issparse(A)):
        raise ValueError("Cannot run sparse affine_null_space on dense matrix")

    method = method.lower()
    if(method == "qr"):
        Q, R, E, rank = spqr(A.T)
        Q = Q.tocsc()
        R = R.tocsc()
        P = permutation_from_E(E).tocsc()

        # Rank of A.T = Row rank of A
        # nc = find(any(abs(R)>tol,2),1,'last');
        nonzero_rows = (abs(R) > tolerance).nonzero()[0]
        if(nonzero_rows.shape[0] > 0):
            nc = nonzero_rows[-1] + 1
        else:
            nc = 0

        # Q  = [Q1, Q2]
        # Q1 = Q(:,1:nc)
        Q1 = Q[:, :nc]
        # Q2 = Q(:,nc+1:end)
        # N  = Q2
        N = Q2 = Q[:, nc:]
        # R = [R1, R2; 0]
        R1 = R[:nc, :nc]

        # A possibly non-unique solution
        # x0 = Q1*(R(1:nc,1:nc)'\(E(:,1:nc)'*(b)))
        b1 = P.T.dot(b)[:nc]
        y1 = scipy.sparse.linalg.spsolve(R1.T, b1)
        x0 = Q1.dot(y1)

    elif(method == "luq"):
        raise NotImplementedError("LUQ decomposition is not implemented!")
        # Special sparse LUQ decomposition
        L, U, Q = luq(A, True, tolerance)
        # Rank
        nonzero_rows = (abs(R) > tolerance).nonzero()[0]
        if(nonzero_rows.shape[0] > 0):
            nc = nonzero_rows[-1] + 1
            mat1 = scipy.sparse.linalg.spsolve(
                U[:nc, :nc], scipy.sparse.eye(nc, L.shape[0]))
            x0 = scipy.sparse.linalg.spsolve(Q, scipy.sparse.vstack(
                [mat1, scipy.sparse.csc_matrix((Q.shape[0] - nc, 1))]))
            QQ = scipy.sparse.linalg.inv(Q)
            N = QQ[:, nc + 1:]
        else:
            m = A.shape[0]
            x0 = numpy.ones((m, 1))
            N = scipy.sparse.identity(m)

    else:
        raise ValueError("Invalild method for solving for the affine null \
            space, %s." % method)

    # return N.multiply(abs(N) >= tolerance), x0
    return N, x0


def affine_null_space(A, b, tolerance = 1e-8, method = "qr", bounds = None):
    """
    Given a system Ax = b, determine a matrix N spanning the right null space
    of A and a feasible solution x0 so that:

        A * (N * y)        = 0  for any y
        A * (N * y) + b    = b
        A * (N * y) + A*x0 = b
        A * (N * y + x0)   = b

        A * (N * y + x0) = b  for any y

    Inputs:
      A - #A by #A (sparse) matrix. Note: I'm pretty sure A must be symmetric
         positive semi-definite.
      b - #A by #b right-hand side
      Options:
        tolerance - tolerance for determine rank
        method - method for finding null space (See dense or sparse for allowed
            values)
        bounds - bounds on x such that bound[0] <= Ny <= bound[1]
            For 0 <= x <= 1:
            (x = Ny + x0 for any y) -> (0 <=  Ny + x0 <= 1)
            -x0 <= Ny <= 1 - x0
            Therefore, bounds = (-x0, 1 - x0)
            NOTE: If type(bounds[i]) == number then
                bounds[i] = numpy.full((#A, #b), bounds[i])
    Outputs:
      N  - #A by #N matrix spanning null space, where #N = #A - rank(A)
      x0 - #A by #b, so that columns are feasible solutions
    """
    # TODO REMOVE THIS
    print(A.shape)

    # Dense matrices -> use dense version
    if(not scipy.sparse.issparse(A)):
        return dense_affine_null_space(A, b, tolerance, method)

    # Use dense version if matrix is smallish
    if(max(A.shape) <= 5000):
        # If dense fails because of method, then try sparse.
        try:
            return dense_affine_null_space(A, b, tolerance, method)
        except ValueError:
            pass

    return sparse_affine_null_space(A, b, tolerance, method)

# Test by generating a singular matrix and running affine_null_space().
if __name__ == "__main__":
    print("%sInputs:%s\n"  % (bcolors.HEADER, bcolors.ENDC))

    M, N = 100, 50 # A is MxN

    # Generate a singular matrix
    # Last row is all zeros
    data = (2 * numpy.random.rand(M, N)).astype("int32")
    data[-1, :] = 0 # sum(data[:-1, :])
    # Make sure the data matrix is singular
    assert M != N or abs(numpy.linalg.det(data)) < 1e-8
    # Convert to a sparse version
    A = scipy.sparse.csc_matrix(data)

    # Generate a b that will always have a solution
    b = A.dot(numpy.ones((N, 1)))

    # Display inputs
    print("A:\n%s\n\nb:\n%s\n" % (A.A, b))

    print("%sOutputs:%s\n" % (bcolors.HEADER, bcolors.ENDC))
    N, x0 = sparse_affine_null_space(A, b, method = "qr")
    print("N:\n%s\n\nx0:\n%s" % (N.A, x0))

    # Ax ?= b
    print("\nAx = \n%s" % A.dot(x0))

    # Test the solution
    num_tests = 1000
    # A * N * y for any y
    print("\nTest A * N * y = 0 for any y:")
    total_diff = sum([
        abs(A.dot(N.dot(numpy.random.rand(N.shape[1], 1)))).sum()
        for i in range(num_tests)])
    test_util.display_results(total_diff < 1e-8, format_str =
        "Expected total: %g, Actual total: %g\t%%s" % (0, total_diff))

    # A * (N * y + x0) = b  for any y
    print("\nTest A * (N * y + x0) = b for any y:")
    total_diff = sum([
        abs(b - (A.dot(N.dot(numpy.random.rand(N.shape[1], 1)) + x0))).sum()
        for i in range(num_tests)])
    test_util.display_results(total_diff < 1e-5, format_str =
        "Expected total: %g, Actual total: %g\t%%s" % (0, total_diff))
