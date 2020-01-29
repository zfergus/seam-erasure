"""
Solves for the energy coefficient matrix, A in x^T*A*x+Bx+C. Energy formula for
bilinear interpolation.

Written by Zachary Ferguson
"""

import itertools

import numpy
import scipy.sparse
import logging
from tqdm import tqdm

from .accumulate_coo import AccumulateCOO
from .seam_intervals import compute_edgePair_intervals
from .util import (lerp_UV, surrounding_pixels,
                   globalEdge_to_local, pairwise, QuadEnergy)


def A_Mat(uv0, uv1, p00, p10, p01, p11, nPixels):
    """ Create a cooefficent A matrix. """
    c1 = (uv1.u - uv0.u) * (uv1.v - uv0.v)
    coeffs = numpy.zeros((1, nPixels))
    coeffs[0, p00] = c1
    coeffs[0, p10] = -c1
    coeffs[0, p01] = -c1
    coeffs[0, p11] = c1
    return coeffs


def B_Mat(uv0, uv1, p00, p10, p01, p11, nPixels):
    """ Create a cooefficent B matrix. """
    c1 = (uv1.u - uv0.u) * uv0.v + uv0.u * (uv1.v - uv0.v)
    c2 = uv1.u - uv0.u
    c3 = uv1.v - uv0.v
    coeffs = numpy.zeros((1, nPixels))
    coeffs[0, p00] = c1 - c2 - c3
    coeffs[0, p10] = -c1 + c2
    coeffs[0, p01] = -c1 + c3
    coeffs[0, p11] = c1
    return coeffs


def C_Mat(uv0, uv1, p00, p10, p01, p11, nPixels):
    """ Create a cooefficent C matrix. """
    c1 = uv0.u * uv0.v
    c2 = uv0.u
    c3 = uv0.v
    c4 = 1
    coeffs = numpy.zeros((1, nPixels))
    coeffs[0, p00] = c1 - c2 - c3 + c4
    coeffs[0, p10] = -c1 + c2
    coeffs[0, p01] = -c1 + c3
    coeffs[0, p11] = c1
    return coeffs


def bilerp_coeffMats(uv0, uv1, p00, p10, p01, p11, nPixels):
    """ Compute the matrix coefficients for a bilinear interpolation. """
    # Compute A, AP
    A = A_Mat(uv0, uv1, p00, p10, p01, p11, nPixels)
    # Compute B, Bp
    B = B_Mat(uv0, uv1, p00, p10, p01, p11, nPixels)
    # Compute C, Cp
    C = C_Mat(uv0, uv1, p00, p10, p01, p11, nPixels)
    return (A, B, C)


def E_ab(a, b, edgePair, width, height):
    """
    Calculate the energy in the inverval a to b.
    Parameters:
        a, b - interval to integrate over
        edgePair - the edge pair to interpolate
        width, height - texture's dimensions
    Returns: Energy matrix for the interval
    """

    # Get the UV lilrdinates of the edge pair, swaping endpoints of one edge
    ((uv0, uv1), (uv1p, uv0p)) = edgePair

    # Determine the midpoint of the interval in UV-space
    mid_uv = lerp_UV((a + b) / 2., uv0, uv1)
    mid_uv_p = lerp_UV((a + b) / 2., uv0p, uv1p)

    # Determine surrounding pixel indices
    p00, p10, p01, p11 = surrounding_pixels(
        mid_uv, width, height, as_index=True)
    p00p, p10p, p01p, p11p = surrounding_pixels(
        mid_uv_p, width, height, as_index=True)

    nPixels = width * height

    luv0, luv1 = globalEdge_to_local(uv0, uv1, p00, width, height)
    luv0p, luv1p = globalEdge_to_local(uv0p, uv1p, p00p, width, height)

    # Compute the coefficient matrix for the interval
    (A, B, C) = bilerp_coeffMats(luv0, luv1, 0, 1, 2, 3, 8)
    (Ap, Bp, Cp) = bilerp_coeffMats(luv0p, luv1p, 4, 5, 6, 7, 8)

    # Each of the A, Ap, B, Bp, C, Cp are 1xN matrices.
    # E is Nx1 * 1xN = NxN
    def term(M, n):
        """
        Compute the integral term with constant matrix (M) and power n after
        integration.
        """
        M *= (1. / n * (b**n - a**n))  # Prevent unnecessary copying
        return M

    # Difference of matrices (1x8)
    Adiff = A - Ap
    Bdiff = B - Bp
    Cdiff = C - Cp

    # Product of differences (8x8)
    AA = Adiff.T.dot(Adiff)
    BB = Bdiff.T.dot(Bdiff)
    CC = Cdiff.T.dot(Cdiff)
    AB = Adiff.T.dot(Bdiff)
    AC = Adiff.T.dot(Cdiff)
    BC = Bdiff.T.dot(Cdiff)

    values = (term(AA, 5.) + term(AB + AB.T, 4.) + term(AC + AC.T + BB, 3.) +
              term(BC + BC.T, 2.) + term(CC, 1.))

    ijs = numpy.array(list(itertools.product(
        (p00, p10, p01, p11, p00p, p10p, p01p, p11p), repeat=2)))

    E = scipy.sparse.coo_matrix((values.ravel(), ijs.reshape(-1, 2).T),
                                shape=(nPixels, nPixels))

    return E


def E_edgePair(edgePair, width, height, edge_len):
    """ Compute the energy coefficient matrix over a single edge pair. """
    intervals = compute_edgePair_intervals(edgePair, width, height)

    N = width * height

    # Space for the matrix.
    # E_edge = scipy.sparse.coo_matrix((N, N))
    E_edge = AccumulateCOO()

    # Solve for the energy coeff matrix over the edge pair
    for a, b in pairwise(intervals):
        # Add intervals energy to total Energy
        # UPDATE: For some reason scipy is converting back and forth to CSR to
        # do the +.
        # E_edge += E_ab(a, b, edgePair, width, height)

        # Grab the guts of each coo matrix.
        E_edge.add(E_ab(a, b, edgePair, width, height))

    # Finally accumulate the total.
    E_edge = E_edge.total((N, N))

    # Multiply by the length of the edge in 3D
    return E_edge * edge_len


def E_total(seam, width, height, depth, edge_lens):
    """ Calculate the energy coeff matrix for a width x height texture. """
    # Sum up the energy coefficient matrices for all the edge pairs
    N = width * height

    # E = scipy.sparse.coo_matrix((N, N))
    E = AccumulateCOO()

    sum_edge_lens = 0.0
    desc = "Building Bilinear Energy Matrix"
    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, (edgePair, edge_len) in enumerate(zip(tqdm(seam, unit="edge pairs",
                                                      desc=desc,
                                                      disable=disable_pbar),
                                                 edge_lens)):
        sum_edge_lens += edge_len
        E.add(E_edgePair(edgePair, width, height, edge_len))

    E = E.total((N, N))

    # Divide by the total edge length in 3D
    return QuadEnergy((E / sum_edge_lens).tocsc(),
                      scipy.sparse.csc_matrix((N, depth)),
                      scipy.sparse.csc_matrix((depth, depth)))
