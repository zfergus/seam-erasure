"""
Seam gradient energy to get a better gradient energy across the seam.

Written by Zachary Ferguson
"""

import itertools
import logging

import numpy
import scipy.sparse
from tqdm import tqdm

from .accumulate_coo import AccumulateCOO
from .seam_intervals import compute_edgePair_intervals
from .util import (is_counterclockwise, lerp_UV, surrounding_pixels,
                   globalEdge_to_local, pairwise, QuadEnergy)

import warnings
warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)


def A_Mat(st_edge, gamma_perp, p00, p10, p01, p11, nPixels):
    """Create a cooefficent matrix A for the equation ApT + Bp."""
    c = gamma_perp[1] * (st_edge[1][0] - st_edge[0][0]) + \
        gamma_perp[0] * (st_edge[1][1] - st_edge[0][1])
    coeffs = numpy.zeros((1, nPixels))
    coeffs[0, p00] = c
    coeffs[0, p10] = -c
    coeffs[0, p01] = -c
    coeffs[0, p11] = c
    return coeffs


def B_Mat(st_edge, gamma_perp, p00, p10, p01, p11, nPixels):
    """Create a cooefficent matrix B for the equation ApT + Bp."""
    c1 = gamma_perp[1] * st_edge[0][0] + gamma_perp[0] * st_edge[0][1]
    c2 = gamma_perp[0]
    c3 = gamma_perp[1]
    coeffs = numpy.zeros((1, nPixels))
    coeffs[0, p00] = c1 - c2 - c3
    coeffs[0, p10] = -c1 + c2
    coeffs[0, p01] = -c1 + c3
    coeffs[0, p11] = c1
    return coeffs


def inside_perpendicular_vector(mesh, edge):
    """
    Returns the normalized vector in the perpendicular inside directions.
    Inputs:
        mesh - the model in OBJ format
        edge - the edge in (fi, (fv0, fv1)) format
    Output:
        Returns the appropriate perpendicular vector pointing inside the UV
        face.
    """
    p0, p1 = [numpy.array(mesh.vt[mesh.f[edge[0]][i].vt]) for i in edge[1]]
    vec = p1 - p0
    if is_counterclockwise(*[mesh.vt[fv.vt] for fv in mesh.f[edge[0]]]):
        perp = numpy.array([-vec[1], vec[0]])
    else:
        perp = numpy.array([vec[1], -vec[0]])
    length = float(numpy.linalg.norm(perp))
    return (perp / length) if (abs(length) > 1e-8) else perp


def E_ab(a, b, mesh, edgePair, width, height):
    """
    Calculate the Energy in the inverval a to b.
    Inputs:
        mesh - the model in OBJ format
        edgePair - the edgePair of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
    Output:
        Returns the energy coefficient matrix for the interval.
    """

    # Get the UV coordinates of the edge pair, swaping endpoints of one edge
    ((uv0, uv1), (uv1p, uv0p)) = [
        [mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]] for edge in edgePair]

    # Determine the midpoint of the interval in UV-space
    mid_uv = lerp_UV((a + b) / 2., uv0, uv1)
    mid_uv_p = lerp_UV((a + b) / 2., uv0p, uv1p)

    # Determine surrounding pixel indices
    (p00, p10, p01, p11) = surrounding_pixels(
        mid_uv, width, height, as_index=True)
    (p00p, p10p, p01p, p11p) = surrounding_pixels(
        mid_uv_p, width, height, as_index=True)

    nPixels = width * height

    st_edge = globalEdge_to_local(uv0, uv1, p00, width, height)
    st_edge_p = globalEdge_to_local(uv0p, uv1p, p00p, width, height)

    perp_edge = inside_perpendicular_vector(mesh, edgePair[0])
    A = A_Mat(st_edge, perp_edge, 0, 1, 2, 3, 8)
    B = B_Mat(st_edge, perp_edge, 0, 1, 2, 3, 8)

    perp_edge_p = inside_perpendicular_vector(mesh, edgePair[1])
    Ap = A_Mat(st_edge_p, perp_edge_p, 4, 5, 6, 7, 8)
    Bp = B_Mat(st_edge_p, perp_edge_p, 4, 5, 6, 7, 8)

    # Each of the A, Ap, B, Bp are 1xN matrices.
    # E is Nx1 * 1xN = NxN
    def term(M, n):
        """
        Compute the integral term with constant matrix (M) and power n after
        integration.
        """
        M *= (1. / n * (b**n - a**n))  # Prevent unnecessary copying
        return M

    # Sum of matrices (1x8)
    Asum = A + Ap
    Bsum = B + Bp

    # Product of sums (8x8)
    AA = Asum.T.dot(Asum)
    BB = Bsum.T.dot(Bsum)
    AB = Asum.T.dot(Bsum)

    values = term(AA, 3.) + term(AB + AB.T, 2.) + term(BB, 1.)

    ijs = numpy.array(list(itertools.product(
        (p00, p10, p01, p11, p00p, p10p, p01p, p11p), repeat=2)))

    # import pdb; pdb.set_trace()

    E = scipy.sparse.coo_matrix(
        (values.ravel(), ijs.reshape(-1, 2).T), shape=(nPixels, nPixels))

    return E


def E_edgePair(mesh, edgePair, width, height, edge_len):
    """
    Compute the energy coefficient matrix over a single edge pair.

    Inputs:
        mesh - the model in OBJ format
        edgePair - the edgePair of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_len - the length of the edge in 3D space

    Output:
        Returns the energy coefficient matrix over a single edge pair.
    """
    uv_edgePair = [[mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
                   for edge in edgePair]
    intervals = compute_edgePair_intervals(uv_edgePair, width, height)

    N = width * height

    # Space for the matrix.
    # E_edge = scipy.sparse.coo_matrix((N, N))
    E_edge = AccumulateCOO()

    # Solve for the energy coeff matrix over the edge pair
    for a, b in pairwise(intervals):
        # Add intervals energy to total Energy
        # UPDATE: For some reason scipy is converting back and forth to CSR
        # to do the +.
        # E_edge = E_edge + E_ab(a, b, mesh, edgePair, width, height)

        # Grab the guts of each coo matrix.
        E_edge.add(E_ab(a, b, mesh, edgePair, width, height))

    # Finally accumulate the total.
    E_edge = E_edge.total((N, N))

    # Multiply by the length of the edge in 3D
    return E_edge * edge_len


def E_total(mesh, seam, width, height, depth, edge_lens):
    """
    Calculate the energy coeff matrix for a width x height texture.

    Inputs:
        mesh - the model in OBJ format
        seam - the seam of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_lens - a list containing the lengths of each edge in 3D space.

    Output:
        Returns the quadtratic term matrix for the seam gradient.
    """
    # Sum up the energy coefficient matrices for all the edge pairs
    N = width * height

    # E = scipy.sparse.coo_matrix((N, N))
    E = AccumulateCOO()

    sum_edge_lens = 0.0

    desc = "Building Seam Gradient Matrix"
    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, (edgePair, edge_len) in enumerate(zip(tqdm(seam, unit="edge pairs",
                                                      desc=desc,
                                                      disable=disable_pbar),
                                                 edge_lens)):
        sum_edge_lens += edge_len
        E.add(E_edgePair(mesh, edgePair, width, height, edge_len))

    E = E.total((N, N))

    # Divide by the total edge length in 3D
    return QuadEnergy((E / sum_edge_lens).tocsc(),
                      scipy.sparse.csc_matrix((N, depth)),
                      scipy.sparse.csc_matrix((depth, depth)))
