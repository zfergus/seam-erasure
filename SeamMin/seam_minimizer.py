"""
Minimize the energy difference of a textures "outside" pixels.

Written by Zachary Ferguson
"""

from __future__ import print_function

import pdb

import os
import sys

from multiprocessing import Process
from recordclass import recordclass

import scipy
import numpy

import obj_reader
from find_seam import find_seam, seam_to_UV
from util import *
import bilerp_energy
import lsq_constraints
from mask import *
from seam_loops import *
import dirichlet
import seam_gradient
import seam_value_energy_texture
import seam_value_energy_lerp
from texture import save_texture
from null_space_method import NullSpaceMethod
import inequalities

energies_str = "BLE, SV, SG, LSQ, LSQ1, LSQ2, L"
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

    bag_of_edges = ([edge for edgepair in seam for edge in edgepair] +
        boundary + foldovers)

    SV = SeamValueMethod.compute_energy(sv_method, mesh, bag_of_edges, width,
        height, textureVec)

    # Find all of the seam loops
    bag_of_edges = ([edge for edgepair in uv_seam for edge in edgepair] +
        uv_boundary + uv_foldovers)

    # Constrain the values
    print("Building Least Squares Constraints")
    lsq_mask = mask_inside_seam(mesh, bag_of_edges, width, height)

    lsq1_mask = mask_seam(mesh, bag_of_edges, width, height)
    LSQ1 = lsq_constraints.constrain_values(lsq1_mask, textureVec)

    lsq2_mask = lsq_mask ^ lsq1_mask
    LSQ2 = lsq_constraints.constrain_values(lsq2_mask, textureVec)

    LSQ = QuadEnergy(LSQ1.Q + LSQ2.Q, LSQ1.L + LSQ2.L, LSQ1.C + LSQ2.C)
    del lsq1_mask, lsq2_mask
    print("Done\n")

    # Construct a dirichlet energy for the texture.
    print("Building Dirichlet Energy")
    dirichlet_mask = mask_inside_faces(
        mesh, width, height, init_mask=~lsq_mask)
    L = dirichlet.dirichlet_energy(height, width, textureVec, ~dirichlet_mask,
        lsq_mask)
    print("Done\n")

    return Energies(BLE=BLE, SV=SV, SG=SG, LSQ=LSQ, LSQ1=LSQ1, LSQ2=LSQ2, L=L)


def solve_seam(mesh, texture, bounds=None, display_energy_file=None,
        method="weighted", sv_method=SeamValueMethod.NONE, do_global=False):
    """
    Solves for the minimized seam values.
    Returns the minimized texture as a numpy array, shape = (N, depth)
    """
    method = method.lower()
    if(method != "nullspace" and method != "weighted"):
        raise ValueError("Invalild method for solving seam values, %s." %
            method)

    # Assert for valid bounds argument
    if(bounds is not None and len(bounds) != 2):
        raise ValueError("Invalid bounds for solving seam values, %s." %
            [bounds])

    height, width, depth = (texture.shape + (1,))[:3]
    N = width * height

    # Get the coefficients for the quadratic energies.
    energies = compute_energies(mesh, texture, sv_method)
    # exec(energies_str + " = energies")
    BLE, SV, SG, LSQ, LSQ1, LSQ2, L = energies # WARNING: Do not change order

    print("Solving for minimal energy solution")
    sys.stdout.flush()

    if(method == "nullspace"):
        # Print progress dots.
        dot_process = Process(target = print_dots)
        # dot_process.start()

        # Lexicographical Ordering of the quadratic energies
        order = [LSQ1, BLE, SV, LSQ2, SG, L]
        H = [] # Quadtratic Terms
        f = [] # Linear Terms
        for E in order:
            if E is not None:
                H.append(E.Q)
                f.append(E.L)

        try:
            solution = NullSpaceMethod(H, f, bounds=bounds).A
        finally:
            # dot_process.terminate()
            pass
    elif(method == "weighted"):
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

        # Should the solution be bounded in the range [a, b] for a, b = bounds
        # TODO: Test the bounding constraints more.
        if(bounds):
            # Weight the Quad and linear terms to be greater than the
            # inequality.
            # TODO: Use a different solver to explain the need for this weight.
            cvxW = 1.0e8 * N
            quad = (cvxW * quad).tocoo()
            lin *= cvxW

            # CVXOPT solver format change.
            P = cvxopt.spmatrix(quad.data, quad.row.astype(int),
                quad.col.astype(int), size=quad.shape)
            q = cvxopt.matrix(lin)

            G, h = inequalities.CVXOPTBoundingMatrix(lin.shape, bounds)

            solution = inequalities.cvxopt_solve_all_depth(
                lin.shape, P, q, G=G, h=h)

        else:
            # Print progress dots.
            dot_process = Process(target = print_dots)
            dot_process.start()

            try:
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

# Quick test of this code; main.py is a cmd-line tool for this file.
if __name__ == "__main__":
    obj = obj_reader.load_obj("cube_edit.obj")
    # obj = obj_reader.load_obj("../models/male_low_poly_edited.obj")
    print("")
    texture = numpy.ones((100, 100))
    print("\nResults: %s" % solve_seam(obj, texture))
