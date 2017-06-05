"""
Convert a tangent space normal map to a global space normal map.
Written by Zachary Ferguson
"""

from __future__ import print_function

import pdb

import numpy

import includes

import texture
import obj_reader
from util import *


def compute_M(uv1, uv2, xyz1, xyz2):
    """
    Compute the transformation from UV to XYZ
    M @ LHS = RHS => M = RHS @ LHS^-1 = (LHS^T^-1 @ RHS^T)^T
    LHS^T @ M^T = RHS^T => M = (LHS^T \ RHS^T)^T
    """
    RHS = numpy.empty((3, 3))
    RHS[:, 0] = xyz1
    RHS[:, 1] = xyz2
    norm = numpy.cross(xyz1, xyz2)
    RHS[:, 2] = norm / numpy.linalg.norm(norm)
    LHS = numpy.zeros((3, 3))
    LHS[:2, 0] = uv1
    LHS[:2, 1] = uv2
    LHS[2, 2] = 1
    return numpy.linalg.solve(LHS.T, RHS.T).T


def tangent_to_global_normals(mesh, texture):
    """
    Convert a tangent space normal map to a global space normal map.
    Input:
        mesh - a mesh in OBJ format
        texture - the tangent space texture map
    Output:
        Returns a global space normal map for the mesh.
    """
    vts = numpy.array(mesh.vt)
    vs = numpy.array(mesh.v)

    # For each face generate a transformation from UV to XYZ
    FaceM = numpy.zeros((len(mesh.f), 3, 3))
    for i, face in enumerate(mesh.f):
        print_progress(i / float(len(mesh.f)))
        uv1 = vts[face[1].vt] - vts[face[0].vt]
        uv2 = vts[face[2].vt] - vts[face[0].vt]
        xyz1 = vs[face[1].v] - vs[face[0].v]
        xyz2 = vs[face[2].v] - vs[face[0].v]
        try:
            M = compute_M(uv1, uv2, xyz1, xyz2)
        except:
            continue
        # Normalize columns of the matrix
        FaceM[i] = M / numpy.sqrt((M * M).sum(axis = 0))
    print_progress(1)
    print()

    width, height = texture.shape[:2]
    out_tex = numpy.zeros((width, height, 3))
    mask = numpy.zeros((height, width), dtype=int)
    for i, face in enumerate(mesh.f):
        print_progress(i / float(len(mesh.f)))
        fvts = numpy.vstack([vts[fi.vt] for fi in face])

        # pdb.set_trace()
        # Compute the bounding box of the face.
        xs = (width * fvts[:, 0]).T # Array of x values inside the bounding box
        ys = (height * fvts[:, 1]).T # y-values
        # Add a border of 1 pixel to the bounding box.
        xleft = max(int(min(xs)) - 3, 0)
        xright = min(int(max(xs)) + 2, width - 1)
        ybottom = max(int(min(ys)) - 3, 0)
        ytop = min(int(max(ys)) + 2, height - 1)

        # B is the transformation from xy to barycentric coordinates
        B = numpy.vstack([xs, ys, numpy.ones(3)])

        grid = numpy.mgrid[xleft:xright + 1, ybottom:ytop + 1].reshape(2, -1)
        grid = numpy.vstack([grid, numpy.ones((1, grid.shape[1]))]).astype(int)

        # Convert the grid from XY pixel locations to barycentric coordinates.
        # This will only fail of the triangle is degenerate.
        try:
            coords = numpy.linalg.solve(B, grid)
        except:
            continue

        # Get the transformation values for the corners as a 3D array
        # corner_vals = numpy.array([FaceM[i] for j in range(len(face))])
        coord_transform = FaceM[i].T

        for j in range(coords.shape[1]):
            bc_coordinates = coords[:, j]
            x, y = grid[:2, j]

            # Global normal for the pixel value
            try:
                # Inverse transpose of the texture
                gNorm = numpy.linalg.solve(coord_transform, texture[y, x, :3])
            except:
                continue

            # Transform the tangent normal
            if numpy.all(coords[:, j] >= -1e-4):
                out_tex[y, x] = gNorm
                mask[y, x] = 2
            elif mask[y, x] == 1:
                out_tex[y, x] += gNorm
                out_tex[y, x] /= 2.0
            elif mask[y, x] == 0:
                out_tex[y, x] = gNorm
                mask[y, x] = 1

    # Normalize the pixels stored in the texture.
    out_tex = out_tex / numpy.sqrt(
        (out_tex * out_tex).sum(axis = 2)).reshape(height, width, 1)

    print_progress(1)
    print()
    return out_tex

if __name__ == "__main__":
    mesh = obj_reader.quads_to_triangles(
        obj_reader.load_obj("../models/cow.obj"))

    data = numpy.array(texture.load_texture(
        "../textures/cow/Cow_Monster_N.png"))
    isFloatTexture = not issubclass(data.dtype.type, numpy.integer)
    if(not isFloatTexture):
        data = data / 255.0
    data = data * 2 - 1

    out_texture = tangent_to_global_normals(mesh, data)
    import util
    # pdb.set_trace()
    out_texture = -out_texture
    out_texture[:, :, 0] *= -1
    out_texture = (out_texture + 1) / 2.0
    pdb.set_trace()
    # out_texture = util.to_uint8(out_texture, normalize = False)
    texture.save_texture_channels(out_texture, "./cow.tiff")
