"""
Solves for the energy coefficient matrix, A in x^T*A*x+Bx+C. Energy formula for
seam value energy.

Written by Zachary Ferguson
"""

import itertools
import logging

import numpy
import scipy.sparse
from tqdm import tqdm

from .bilerp_energy import bilerp_coeffMats

from .seam_intervals import compute_edge_intervals
from .accumulate_coo import AccumulateCOO
from .util import (lerp_UV, surrounding_pixels,
                   globalEdge_to_local, pairwise, QuadEnergy)

import warnings
warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)


def E_ab(a, b, mesh, edge, width, height):
    """
    Calculate the energy in the inverval a to b.

    Parameters
    ----------
    a, b - interval to integrate over
    mesh - 3D model in OBJ format
    edge - the edge in (fi, (fv0, fv1)) format
    width, height - texture's dimensions

    Returns
    -------
    Energy matrix for the interval

    """
    # Get the UV coordinates of the edge pair, swaping endpoints of one edge
    uv0, uv1 = [mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
    x0, x1 = [
        numpy.array(mesh.vc[mesh.f[edge[0]][i].v]).reshape(1, -1)
        for i in edge[1]
    ]

    # Determine the midpoint of the interval in UV-space
    mid_uv = lerp_UV((a + b) / 2., uv0, uv1)

    # Determine surrounding pixel indices
    (p00, p10, p01, p11) = surrounding_pixels(
        mid_uv, width, height, as_index=True)

    nPixels = width * height

    luv0, luv1 = globalEdge_to_local(uv0, uv1, p00, width, height)

    # Compute the coefficient matrix for the interval
    (A, B, C) = bilerp_coeffMats(luv0, luv1, 0, 1, 2, 3, 4)

    # Each of the A, Ap, B, Bp, C, Cp are 1xN matrices.
    # Q is Nx1 * 1xN = NxN
    def term(M, n):
        """
        Compute the integral term with constant matrix (M) and power n after
        integration.
        """
        M *= (1. / n * (b**n - a**n))  # Prevent unnecessary copying
        return M

    # Product of cooefficents (4x4)
    AA = A.T.dot(A)
    AB = A.T.dot(B)
    AC = A.T.dot(C)
    BB = B.T.dot(B)
    BC = B.T.dot(C)
    CC = C.T.dot(C)

    values = (term(AA, 5.) + term(AB + AB.T, 4.) + term(AC + AC.T + BB, 3.)
              + term(BC + BC.T, 2.) + term(CC, 1.))

    ijs = numpy.array(list(itertools.product((p00, p10, p01, p11), repeat=2)))

    Q = scipy.sparse.coo_matrix(
        (values.ravel(), ijs.reshape(-1, 2).T), shape=(nPixels, nPixels))

    # Difference in endpoints
    x1_x0 = x1 - x0

    # A, B, C are 1xN and x0, x1 are 1xD
    # L is Nx1 * 1xD = NxD
    values = (term(A.T.dot(x1_x0), 4.0)
              + term(A.T.dot(x0) + B.T.dot(x1_x0), 3.0)
              + term(B.T.dot(x0) + C.T.dot(x1_x0), 2.0)
              + term(C.T.dot(x0), 1.0))

    ijs = numpy.array(list(itertools.product(
        (p00, p10, p01, p11), range(x0.shape[1]))))

    L = scipy.sparse.coo_matrix(
        (values.ravel(), ijs.reshape(-1, 2).T), shape=(nPixels, x0.shape[1]))

    # x0, x1 are 1xD
    # C is Dx1 * 1xD = DxD
    x1_x0x0 = x1_x0.T.dot(x0)

    C = (term(x1_x0.T.dot(x1_x0), 3.0) + term(x1_x0x0 + x1_x0x0.T, 2.0)
         + term(x0.T.dot(x0), 1.0))

    return Q, L, C


def E_edge(mesh, edge, width, height, edge_len):
    """
    Compute the energy coefficient matrix over a single edge.

    Inputs:
        mesh - the model in OBJ format
        edge - the edge in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_len - the length of the edge in 3D space
    Output:
        Returns the energy coefficient matrix over a single edge.
    """
    uv_edge = [mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
    intervals = sorted(list(compute_edge_intervals(uv_edge, width, height)))

    N = width * height
    depth = len(mesh.vc[0])

    Q_edge = AccumulateCOO()
    L_edge = AccumulateCOO()
    C_edge = scipy.sparse.csc_matrix((depth, depth))

    # Solve for the energy coeff matrix over the edge pair
    for a, b in pairwise(intervals):
        # Add intervals energy to total Energy
        Q, L, C = E_ab(a, b, mesh, edge, width, height)
        Q_edge.add(Q)
        L_edge.add(L)
        C_edge += C

    Q_edge = Q_edge.total((N, N))
    L_edge = L_edge.total((N, depth))

    # Multiply by the length of the edge in 3D
    return edge_len * Q_edge, edge_len * L_edge, edge_len * C_edge


def E_total(mesh, edges, width, height):
    """
    Calculate the energy coeff matrix for a width x height texture.

    Inputs:
        mesh - the model in OBJ format
        edges - edges of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
    Assume:
        depth == len(mesh.vc[0])
    Output:
        Returns the quadtratic term matrix for the seam value energy.
    """
    # Check the model contains vertex colors.
    if (len(mesh.vc) != len(mesh.v)):
        raise ValueError(
            "Mesh does not contain an equal number vertex colors "
            "and vertices.")

    # Sum up the energy coefficient matrices for all the edge pairs
    N = width * height
    depth = len(mesh.vc[0])

    Q = AccumulateCOO()
    L = AccumulateCOO()
    C = scipy.sparse.csc_matrix((depth, depth))

    sum_edge_lens = 0.0
    desc = "Building Seam Value of Lerp Energy Matrix"
    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, edge in enumerate(tqdm(edges, unit="edges", disable=disable_pbar,
                                  desc=desc)):
        # Calculate the 3D edge length
        verts = [numpy.array(mesh.v[mesh.f[edge[0]][i].v]) for i in edge[1]]
        edge_len = numpy.linalg.norm(verts[1] - verts[0])
        sum_edge_lens += edge_len

        # Compute the QuadEnergy of the edge.
        Qe, Le, Ce = E_edge(mesh, edge, width, height, edge_len)
        # Q += Qe
        Q.add(Qe)
        # L += Le
        L.add(Le)
        C += Ce

    Q = Q.total((N, N))
    L = L.total((N, depth))

    # Divide by the total edge length in 3D
    return QuadEnergy(
        (Q / sum_edge_lens).tocsc(), -L / sum_edge_lens, C / sum_edge_lens)
