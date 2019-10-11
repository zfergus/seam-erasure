"""
Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)

Let the energy E = | G*x - U |^2_M
where:
    x is a row-by-col image flattened into a vector with row-by-col entries.
    G is the gradient operator,
    U are the target gradients (possibly U = G*y for some known function y),
    and M is the diagonal mass matrix which is the norm of each derivative in
    the summation.
Then E = (G*x - U).T * M * (G*x - U) = x.T*G.T*M*G*x + U.T*M*U -
            2*x.T*G.T*M*U
Let L = G.T*M*G.
Note that G has a row for each row and column derivative in the image.
Let S be a matrix that zeros gradients resulting from the application of G
    (e.g. G*y).
Formally, let S be an identity matrix whose dimensions are the same as G has
rows, modified to have entries corresponding to the derivatives involving some
grid positions to be zero. Note that S = S*S and S.T = S and S*M = M*S.
Let U = S*G*y for some known other image y, and S has zeros for values in y we
don't trust.
Then
E = x.T*L*x + y.T*G.T*S.T*M*S*G*y - 2*x.T*G.T*M*S*G*y
  = x.T*L*x + y.T*G.T*S*M*S*G*y - 2*x.T*G.T*S*M*S*G*y
  = x.T*L*x + y.T*L'*y - 2*x.T*L'*y
where L' = G.T*S*M*S*G = G.T*M*S*G = G.T*S*M*G.
The derivative of 1/2 E with respect to x is:
    0.5 * dE/dx = L*x - L'*y
so the minimizing x could be found by solving
    L*x = L'*y

The G, M, and S in the above description are exactly what is returned by
grad_and_mass() in this module.
mask should be True for every grid location we care about (want a solution to).
skip should be True for every grid location we have a known good value for.
"""

from __future__ import division

import logging

import numpy
from scipy import sparse

from .util import QuadEnergy


def grad_and_mass(rows, cols, mask=None, skip=None):
    """
    Returns a gradient operator matrix G for a grid of dimensions
        'rows' by 'cols',
    a corresponding mass matrix M such that L = G.T*M*G is a
    Laplacian operator matrix that is symmetric and normalized such that the
    diagonal values of interior vertices equal 1,
    and a skip matrix S such that the gradient entries of S*G*x are zero for
    (i,j) such that skip[i,j] is False. If skip is None, all entries are
    assumed to be True and S will be the identity matrix.

    Optional parameter `mask` will result in a gradient operator that entirely
    ignores (i,j) such that mask[i,j] is False.

    In other words, `mask` should be True for every grid location
    you care about (want a solution to via L = G.T*M*G).
    `skip` should be True for every grid location you have a known good value
    for.

    Matrices returned are scipy.sparse matrices.
    """

    assert rows > 0 and cols > 0

    if mask is not None:
        mask = numpy.asarray(mask, dtype=bool)
        assert mask.shape == (rows, cols)

    if skip is not None:
        skip = numpy.asarray(skip, dtype=bool)
        assert skip.shape == (rows, cols)

    # The number of derivatives in the +row direction is cols * (rows - 1),
    # because the bottom row doesn't have them.
    num_Grow = cols * (rows - 1)
    # The number of derivatives in the +col direction is rows * (cols - 1),
    # because the right-most column doesn't have them.
    num_Gcol = rows * (cols - 1)

    # Gradient matrix
    gOnes = numpy.ones(num_Grow + num_Gcol)
    vals = numpy.append(-gOnes, gOnes)
    del gOnes

    gColRange = numpy.arange(rows * cols)
    gColRange = gColRange[~(gColRange % cols == (cols - 1))]
    colJ = numpy.concatenate([
        numpy.arange(num_Grow),
        gColRange,
        numpy.arange(cols, num_Grow + cols),
        gColRange + 1])
    del gColRange

    # Skip matrix
    if(skip is not None):
        S_diag = numpy.append(
            skip[:-1] & skip[1:], skip[:, :-1] & skip[:, 1:]).astype(int)
    else:
        S_diag = numpy.ones(num_Grow + num_Gcol)

    # Mass diagonal matrix
    if(mask is not None):
        m = numpy.zeros((rows - 1, cols))
        m[:, 1:][mask[:-1, :-1] & mask[1:, :-1]] += 0.125
        m[:, :-1][mask[:-1, 1:] & mask[1:, 1:]] += 0.125
        mass = m.flatten()
        m = numpy.zeros((rows, cols - 1))
        m[1:][mask[:-1, :-1] & mask[:-1, 1:]] += 0.125
        m[:-1][mask[1:, :-1] & mask[1:, 1:]] += 0.125
        mass = numpy.append(mass, m)
    else:
        m = numpy.hstack([numpy.full((rows - 1, 1), 0.125),
                          numpy.full((rows - 1, cols - 2), 0.25),
                          numpy.full((rows - 1, 1), 0.125)])
        mass = m.flatten()
        m = numpy.vstack([numpy.full((1, cols - 1), 0.125),
                          numpy.full((rows - 2, cols - 1), 0.25),
                          numpy.full((1, cols - 1), 0.125)])
        mass = numpy.append(mass, m.flatten())
    del m
    output_row = num_Grow + num_Gcol

    if(mask is not None):
        keep_rows = numpy.append(mask[:-1] & mask[1:],
                                 mask[:, :-1] & mask[:, 1:])
        tiled_keep_rows = numpy.tile(keep_rows, 2)
        vals = vals[tiled_keep_rows]
        colJ = colJ[tiled_keep_rows]
        S_diag = S_diag[keep_rows]
        mass = mass[keep_rows]
        output_row = numpy.count_nonzero(keep_rows)

    # rowI is dependent on the number of output rows.
    rowI = numpy.tile(numpy.arange(output_row), 2)

    G = sparse.coo_matrix((vals, (rowI, colJ)),
                          shape=(output_row, rows * cols))
    assert G.shape == (output_row, rows * cols)

    M = coo_diag(mass)
    assert M.shape == (output_row, output_row)

    S = coo_diag(S_diag)
    assert S.shape == (output_row, output_row)

    return G, M, S


def gen_symmetric_grid_laplacian(rows, cols, mask=None):
    """
    Returns a Laplacian operator matrix for a grid of dimensions 'rows' by
    'cols'. The matrix is symmetric and normalized such that the diagonal
    values of interior vertices equal 1.

    Matrices returned are scipy.sparse matrices.
    """

    assert rows > 0 and cols > 0

    G, M, S = grad_and_mass(rows, cols, mask)
    return G.T * M * G


def dirichlet_energy(rows, cols, y, mask=None, skip=None):
    """
    Builds the quadratic energy for the Dirichlet.

        E  = x.T*L*x + y.T*G.T*S.T*M*S*G*y - 2*x.T*G.T*M*S*G*y
           = x.T*L*x + y.T*G.T*S*M*S*G*y - 2*x.T*G.T*S*M*S*G*y
           = x.T*L*x + y.T*L'*y - 2*x.T*L'*y
        L  = G.T*M*G
        L' = G.T*S*M*G

    Inputs:
        (rows, cols) - deminsions of texture/2D grid
        y - Vector of original value (y above).
        mask - Ignore values cooresponding to False
        skip - True for good prexisting values
    Output:
        A QuadEnergy object containing the values for the quadratic, linear,
        and constant terms.
    """
    assert rows > 0 and cols > 0

    G, M, S = grad_and_mass(rows, cols, mask, skip)
    G, M, S = G.tocsc(), M.tocsc(), S.tocsc()

    L = (G.T.dot(M.dot(G)))
    Lp = G.T.dot(S.dot(M.dot(G)))

    return QuadEnergy(L, -sparse.csc_matrix(Lp.dot(y)),
                      sparse.csc_matrix(y.T.dot(Lp.dot(y))))


def coo_diag(vals):
    try:
        indices = numpy.arange(vals.shape[0])
    except Exception:
        indices = numpy.arange(len(vals))
    return sparse.coo_matrix((vals, (indices, indices)))


def test_mask():
    logging.info('=== test_mask() ===')

    shape = (5, 5)

    mask = numpy.ones(shape, dtype=bool)
    mask[2, 2] = False

    G, M, S = grad_and_mass(shape[0], shape[1], mask=mask)
    L = G.T * M * G

    # set_printoptions(linewidth = 200)
    # print(8*L.toarray())
    # Print the weight of each grid vertex.
    # The weight should be the valence (number of neighbors)/4.
    # Multiply by 4 to get whole numbers
    logging.info(4 * L.diagonal().reshape(shape))


if __name__ == '__main__':
    from seam_erasure import display_quadratic_energy

    # test_mask()
    sizes = [(4, 4, 1), (10, 10, 1), (100, 100, 1), (1000, 1000, 1)]
    for size in sizes:
        logging.info("Texture Size: %s" % (size, ))
        width, height, depth = size
        N = width * height

        diriTex = numpy.linspace(0, 1, width)
        diriTex = numpy.tile(
            numpy.repeat(diriTex, depth).reshape((1, width, depth)),
            (width, 1, 1))
        diriTex = diriTex.reshape(N, -1)
        inTex = numpy.zeros((N, depth))
        # import pdb; pdb.set_trace()
        coeff = dirichlet_energy(height, width, inTex)
        display_quadratic_energy(coeff, inTex, diriTex, "Dirichlet")
