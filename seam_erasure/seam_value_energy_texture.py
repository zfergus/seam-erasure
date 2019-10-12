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


def E_ab(a, b, edge, width, height):
    """
    Calculate the energy in the inverval a to b.

    Parameters
    ----------
    a, b - interval to integrate over
    edge - the edge in UV-space to interpolate
    width, height - texture's dimensions

    Returns
    -------
    Energy matrix for the interval

    """

    # Get the UV coordinates of the edge pair, swaping endpoints of one edge
    uv0, uv1 = edge

    # Determine the midpoint of the interval in UV-space
    mid_uv = lerp_UV((a + b) / 2., uv0, uv1)

    # Determine surrounding pixel indices
    p00, p10, p01, p11 = surrounding_pixels(
        mid_uv, width, height, as_index=True)

    nPixels = width * height

    luv0, luv1 = globalEdge_to_local(uv0, uv1, p00, width, height)

    # Compute the coefficient matrix for the interval
    (A, B, C) = bilerp_coeffMats(luv0, luv1, 0, 1, 2, 3, 4)

    # Each of the A, Ap, B, Bp, C, Cp are 1xN matrices.
    # E is Nx1 * 1xN = NxN
    def term(M, n):
        """
        Compute the integral term with constant matrix (M) and power n after
        integration.
        """
        M *= (1. / n * (b**n - a**n))  # Prevent unnecessary copying
        return M

    # Product of differences (8x8)
    AA = A.T.dot(A)
    AB = A.T.dot(B)
    AC = A.T.dot(C)
    BB = B.T.dot(B)
    BC = B.T.dot(C)
    CC = C.T.dot(C)

    values = (term(AA, 5.) + term(AB + AB.T, 4.) + term(AC + AC.T + BB, 3.)
              + term(BC + BC.T, 2.) + term(CC, 1.))

    ijs = numpy.array(list(itertools.product((p00, p10, p01, p11), repeat=2)))

    E = scipy.sparse.coo_matrix(
        (values.ravel(), ijs.reshape(-1, 2).T), shape=(nPixels, nPixels))

    return E


def E_edge(edge, width, height, edge_len):
    """Compute the energy coefficient matrix over a single edge pair."""
    intervals = sorted(list(compute_edge_intervals(edge, width, height)))

    N = width * height

    # Space for the matrix.
    # E_edge = scipy.sparse.coo_matrix((N, N))
    E_edge = AccumulateCOO()

    # Solve for the energy coeff matrix over the edge pair
    for a, b in pairwise(intervals):
        # Add intervals energy to total Energy
        # E_edge += E_ab(a, b, edge, width, height)
        E_edge.add(E_ab(a, b, edge, width, height))

    # Finally accumulate the total.
    E_edge = E_edge.total((N, N))

    # Multiply by the length of the edge in 3D
    return E_edge * edge_len


def E_total(mesh, edges, width, height, textureVec):
    """
    Calculate the energy coeff matrix for a width x height texture.

    Inputs:
        mesh - the model in OBJ format
        edges - edges of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        textureVec - (width*height)x(depth) vector for the texture
    Output:
        QuadEnergy for the total seam value energy.
    """
    # Sum up the energy coefficient matrices for all the edge pairs
    N = width * height

    # E = scipy.sparse.coo_matrix((N, N))
    E = AccumulateCOO()

    sum_edge_lens = 0.0
    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, edge in enumerate(tqdm(edges, unit="edges", disable=disable_pbar,
                                  desc="Building Seam Value Energy Matrix")):
        face = mesh.f[edge[0]]
        # Calculate the 3D edge length
        verts = [numpy.array(mesh.v[face[i].v]) for i in edge[1]]
        edge_len = numpy.linalg.norm(verts[1] - verts[0])
        sum_edge_lens += edge_len
        # Convert to UV edge
        uv_edge = [mesh.vt[face[i].vt] for i in edge[1]]
        # Calculate the energy of the edge
        E.add(E_edge(uv_edge, width, height, edge_len))

    E = E.total((N, N))

    # Divide by the total edge length in 3D
    SV = (E / sum_edge_lens).tocsc()
    p0 = textureVec
    return QuadEnergy(SV, scipy.sparse.csc_matrix(-SV.dot(p0)),
                      scipy.sparse.csc_matrix(p0.T.dot(SV.dot(p0))))
