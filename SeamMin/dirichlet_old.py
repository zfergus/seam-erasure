"""
Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)

Compute the Dirichlet Laplacian.
"""

from scipy import *
from numpy import *


def cut_edges_from_mask(mask):
    """
    Given a boolean 2D array 'mask', returns 'cut_edges' suitable for passing
    to 'gen_symmetric_grid_laplacian()' where the horizontal and vertical
    transitions between True and False entries becomes cut edges.

    tested
    (see also test_cut_edges())
    """

    mask = asarray(mask, dtype = bool)

    cut_edges = []

    horiz = mask[:-1, :] != mask[1:, :]
    for i, j in zip(*where(horiz)):
        cut_edges.append(((i, j), (i + 1, j)))
    del horiz

    vert = mask[:, :-1] != mask[:, 1:]
    for i, j in zip(*where(vert)):
        cut_edges.append(((i, j), (i, j + 1)))
    del vert

    return cut_edges


def assert_valid_cut_edges(rows, cols, cut_edges):
    """ Check that the cut edges are valid. """
    assert rows > 0 and cols > 0

    if len(cut_edges) > 0:
        # cut edge indices must both be inside the grid
        def in_grid(i, j):
            return i >= 0 and i < rows and j >= 0 and j < cols
        assert all([in_grid(i, j) and in_grid(k, l) for (i, j), (k, l) in
            cut_edges])
        # cut edges must be horizontal or vertical neighbors
        assert all([abs(i - k) + abs(j - l) == 1 for (i, j), (k, l) in
            cut_edges])
        # cut edges must be unique
        assert len(frozenset([(tuple(ij), tuple(kl)) for ij, kl in
            cut_edges])) == len(cut_edges)


def gen_symmetric_grid_laplacian2(rows, cols, cut_edges = None):
    '''
    Returns a Laplacian operator matrix for a grid of dimensions 'rows' by
    'cols'. The matrix is symmetric and normalized such that the diagonal
    values of interior vertices equal 1.

    Optional parameter 'cut_edges', a sequence of 2-tuples ((i,j), (k,l))
    where (i,j) and (k,l) must be horizontal or vertical neighbors in the grid,
    specifies edges in the grid which should be considered to be disconnected.

    tested
    (see also test_cut_edges())
    '''

    assert rows > 0
    assert cols > 0

    if cut_edges is None:
        cut_edges = []

    assert_valid_cut_edges(rows, cols, cut_edges)

    from scipy import sparse

    N = rows
    M = cols

    def ind2ij(ind):
        assert ind >= 0 and ind < N * M
        return ind // M, ind % M

    def ij2ind(i, j):
        assert i >= 0 and i < N and j >= 0 and j < M
        return i * M + j

    Adj = []
    AdjValues = []

    # The middle (lacking the first and last columns) strip down
    # to the bottom, not including the bottom row.
    for i in range(0, rows - 1):
        for j in range(1, cols - 1):

            ind00 = ij2ind(i, j)
            indp0 = ij2ind(i + 1, j)
            Adj.append((ind00, indp0))
            AdjValues.append(.25)

    # The first and last columns down to the bottom,
    # not including the bottom row.
    for i in range(0, rows - 1):
        for j in (0, cols - 1):

            ind00 = ij2ind(i, j)
            indp0 = ij2ind(i + 1, j)
            Adj.append((ind00, indp0))
            AdjValues.append(.125)

    # The middle (lacking the first and last rows) strip to
    # the right, not including the last column.
    for i in range(1, rows - 1):
        for j in range(0, cols - 1):

            ind00 = ij2ind(i, j)
            ind0p = ij2ind(i, j + 1)
            Adj.append((ind00, ind0p))
            AdjValues.append(.25)

    # The first and last rows over to the right,
    # not including the right-most column.
    for i in (0, rows - 1):
        for j in range(0, cols - 1):

            ind00 = ij2ind(i, j)
            ind0p = ij2ind(i, j + 1)
            Adj.append((ind00, ind0p))
            AdjValues.append(.125)

    # Build the adjacency matrix.
    AdjMatrix = sparse.coo_matrix((AdjValues, asarray(Adj).T),
        shape = (rows * cols, rows * cols))
    # We have so far only counted right and downward edges.
    # Add left and upward edges by adding the transpose.
    AdjMatrix = AdjMatrix.T + AdjMatrix
    # AdjMatrix = AdjMatrix.tocsc()

    # Build the adjacency matrix representing cut edges and subtract it
    if len(cut_edges) > 0:
        CutAdj = []
        for ij, kl in cut_edges:
            CutAdj.append((ij2ind(*ij), ij2ind(*kl)))
            CutAdj.append((ij2ind(*kl), ij2ind(*ij)))
        CutAdjMatrix = sparse.coo_matrix((ones(len(CutAdj)),
            asarray(CutAdj).T), shape = (rows * cols, rows * cols))

        # Update AdjMatrix.
        # We need to subtract the component-wise product of CutAdjMatrix
        # and AdjMatrix because AdjMatrix has non-zero values and CutAdjMatrix
        # acts like a mask.
        AdjMatrix = AdjMatrix - CutAdjMatrix.multiply(AdjMatrix)

    '''
    # One over mass
    ooMass = sparse.identity(rows*cols)
    ooMass.setdiag(1./asarray(AdjMatrix.sum(1)).ravel())
    # NOTE: ooMass*AdjMatrix isn't symmetric because of boundaries!!!
    L = sparse.identity(rows*cols) - ooMass * AdjMatrix
    '''

    # This formulation is symmetric: each vertex has a consistent weight
    # according to its area (meaning boundary vertices have smaller
    # weights than interior vertices).
    # NOTE: I tried sparse.dia_matrix(), but sparse.dia_matrix.setdiag() fails
    # with a statement that dia_matrix doesn't have element assignment.
    # UPDATE: setdiag() seems to just be generally slow.  coo_matrix is fast!
    # Mass = sparse.lil_matrix((rows*cols, rows*cols))
    # Mass.setdiag(asarray(AdjMatrix.sum(1)).ravel())
    # debugger()
    Mass = sparse.coo_matrix((asarray(AdjMatrix.sum(1)).ravel(),
        (range(rows * cols), range(rows * cols))))
    L = (Mass - AdjMatrix)

    # The rows should sum to 0.
    assert (abs(asarray(L.sum(1)).ravel()) < 1e-5).all()
    # The columns should also sum to 0, since L is symmetric.
    assert (abs(asarray(L.sum(0)).ravel()) < 1e-5).all()
    # It should be symmetric.
    assert len((L - L.T).nonzero()[0]) == 0

    return L


# L = gen_symmetric_grid_laplacian2(rows, cols, cut_edges_from_mask(mask))
