"""
Minimize the energy difference of a textures "outside" pixels.

Written by Zachary Ferguson
"""

from __future__ import print_function

import sys

from multiprocessing import Process
from recordclass import recordclass

import scipy
import numpy

import obj_reader
from find_seam import find_seam, seam_to_UV
import bilerp_energy
import lsq_constraints
from mask import *
import dirichlet
import seam_gradient
import seam_value_energy_texture
import seam_value_energy_lerp
from texture import save_texture
from util import *

energies_str = "BLE, SV, SG, LSQ, L"
Energies = recordclass("Energies", energies_str)


class SeamValueMethod:
    """ Enum for the seam value methods. """
    NONE    = 0
    TEXTURE = 1
    LERP    = 2

    @staticmethod
    def compute_energy(method, mesh, edges, width, height, textureVec):
        if(method == SeamValueMethod.NONE):
            print("!!! Not using Seam Value Energy !!!\n")
            return None
        elif(method == SeamValueMethod.TEXTURE):
            return seam_value_energy_texture.E_total(
                mesh, edges, width, height, textureVec)
        elif(method == SeamValueMethod.LERP):
            return seam_value_energy_lerp.E_total(mesh, edges, width, height)
        else:
            raise ValueError("Invalid method of computing Seam Value Energy.")

    @staticmethod
    def get_name(method):
        if(method == NONE):
            return "No_Seam_Value_Energy"
        elif(method == TEXTURE):
            return "Seam_Value_Texture_Energy"
        elif(method == LERP):
            return "Seam_Value_Lerp_Energy"
        else:
            raise ValueError("Invalid seam value method, %s." % method)


def display_quadratic_energy(coeffs, x0, x, name, out=sys.stdout):
    """ Compute the energy of a solution given the coefficents. """
    print("%s Before After" % name)
    E0 = x0.T.dot(coeffs.Q.dot(x0)) + 2.0 * x0.T.dot(coeffs.L.A) + coeffs.C.A
    E = x.T.dot(coeffs.Q.dot(x)) + 2.0 * x.T.dot(coeffs.L.A) + coeffs.C.A
    depth = (x.shape + (1,))[1]
    for i in range(depth):
        print("%d %g %g" % (i, E0[i] if depth < 2 else E0[i, i],
            E[i] if depth < 2 else E[i, i]))


def display_energies(energies, x0, x, out=sys.stdout):
    """
    Display the bilinear and Dirichlet energies.
    Inputs:
        energies - an Energies object for the coefficents of the quadratic
            energies.
        x0 - original vector
        x - solution vector
    """
    # LSQ = QuadEnergy(2 * energies.LSQ.Q, energies.LSQ.L, energies.LSQ.C)
    names = ["Bilinear_Energy", "Seam_Value_Energy", "Seam_Gradient_Energy",
        "Least_Squares_Energy", "Dirichlet_Energy"]
    coeffs = [energies.BLE, energies.SV, energies.SG, energies.LSQ, energies.L]
    for name, energy in zip(names, coeffs):
        if(energy):
            display_quadratic_energy(energy, x0, x, name, out=out)
        else:
            print("%s\nN/a" % name)
        print("")


def compute_seam_lengths(mesh, seam):
    """ Calculate the length, in 3D, of all edges on the seam. """
    lens = []
    for edgePair in seam:
        fi, (fv0, fv1) = edgePair[0]
        v0 = numpy.array(mesh.v[mesh.f[fi][fv0].v])
        v1 = numpy.array(mesh.v[mesh.f[fi][fv1].v])
        lens.append(numpy.linalg.norm(v1 - v0))
    return lens


def compute_energies(mesh, texture, sv_method=SeamValueMethod.NONE):
    """
        Minimize the difference between the values of cooresponding edges,
        edge pairs.
        Parameters:
            mesh - a OBJ recordclass for the mesh
            texture - a height x width x depth numpy array of texture values
        Returns:
            Returns a Energies object containing the coefficents for the
            quadtratic energies.
    """

    height, width, depth = (texture.shape + (1,))[:3]
    N = width * height
    textureVec = texture.reshape(N, -1)

    print("Finding seam of model")
    seam, boundary, foldovers = find_seam(mesh)
    uv_seam, uv_boundary, uv_foldovers = seam_to_UV(
        mesh, seam, boundary, foldovers)
    print("Done\n")

    print("Number of edges along the seam: %d" % (len(seam) * 2))
    print("Number of edges along the boundary: %d" % len(boundary))
    print("Number of foldover edges: %d\n" % len(foldovers))

    print("Computing seam edge lengths")
    edge_lens = compute_seam_lengths(mesh, seam)
    print("Done\n")

    # Calculate the energy coeff matrix
    BLE = bilerp_energy.E_total(uv_seam, width, height, depth, edge_lens)

    SG = seam_gradient.E_total(mesh, seam, width, height, depth, edge_lens)

    bag_of_F_edges = ([edge for edgepair in seam for edge in edgepair] +
        boundary + foldovers)

    SV = SeamValueMethod.compute_energy(sv_method, mesh, bag_of_F_edges, width,
        height, textureVec)

    # All edges unsorted
    bag_of_UV_edges = ([edge for edgepair in uv_seam for edge in edgepair] +
        uv_boundary + uv_foldovers)

    # Constrain the values
    print("Building Least Squares Constraints")
    lsq_mask = mask_inside_seam(mesh, bag_of_UV_edges, width, height)
    LSQ = lsq_constraints.constrain_values(lsq_mask, textureVec)
    print("Done\n")

    # Construct a dirichlet energy for the texture.
    print("Building Dirichlet Energy")
    dirichlet_mask = mask_inside_faces(mesh, width, height,
        init_mask=~lsq_mask)
    L = dirichlet.dirichlet_energy(height, width, textureVec, ~dirichlet_mask,
        lsq_mask)
    print("Done\n")

    return Energies(BLE=BLE, SV=SV, SG=SG, LSQ=LSQ, L=L)


def solve_seam(mesh, texture, display_energy_file=None,
        sv_method=SeamValueMethod.NONE, do_global=False):
    """
    Solves for the minimized seam values.
    Returns the minimized texture as a numpy array, shape = (N, depth)
    """
    height, width, depth = (texture.shape + (1,))[:3]
    N = width * height

    # Get the coefficients for the quadratic energies.
    energies = compute_energies(mesh, texture, sv_method)
    BLE, SV, SG, LSQ, L = energies # WARNING: Do not change orde

    print("Solving for minimal energy solution")
    sys.stdout.flush()

    # Minimize energy with constraints (Quad * x = lin)
    # Weights in the order [bleW, svW, sgW, lsqW, diriW]
    if(do_global):
        weights = 1e10, 1e2, 1e2, 1e2, 1e0
    else:
        weights = 1e10, 1e2, 1e2, 1e4, 1e0

    quad = scipy.sparse.csc_matrix((N, N)) # Quadratic term
    lin = scipy.sparse.csc_matrix((N, depth)) # Linear term
    for weight, E in zip(weights, [BLE, SV, SG, LSQ, L]):
        if E is not None:
            quad += weight * E.Q
            lin += weight * E.L

    # Print progress dots.
    dot_process = Process(target = print_dots)
    dot_process.start()

    try:
        # CVXOPT cholmod linsolve should be less memory intensive.
        import cvxopt
        import cvxopt.cholmod
        quad = quad.tocoo()
        system = cvxopt.spmatrix(quad.data, numpy.array(quad.row, dtype=int),
            numpy.array(quad.col, dtype=int))
        rhs = cvxopt.matrix(-lin.A)
        cvxopt.cholmod.linsolve(system, rhs)
        solution = numpy.array(rhs)
    except Exception as e:
        print('cvxopt.cholmod failed,' +
            'using scipy.sparse.linalg.spsolve(): %s' % e)
        solution = scipy.sparse.linalg.spsolve(quad, -lin)
    finally:
        dot_process.terminate()

    print("Done\n")

    if(display_energy_file):
        display_energies(energies, texture.reshape(N, -1), solution,
            out=display_energy_file)

    if scipy.sparse.issparse(solution):
        return solution.A
    return solution
