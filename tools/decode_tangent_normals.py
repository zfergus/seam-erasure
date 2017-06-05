"""
Convert a tangent space normal map to a global space normal map.
Written by Zachary Ferguson
"""

from __future__ import print_function

import pdb

import os
import argparse
import time

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


def compute_vt_transforms(mesh):
    """
    Computes the tranformation matricies from the texture vetices to 3D
    vertices.
    """
    vts = numpy.array(mesh.vt)
    vs = numpy.array(mesh.v)

    # For each face generate a transformation from UV to XYZ
    vtM = numpy.zeros((len(mesh.vt), 3, 3))
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
        for vti in [fi.vt for fi in face]:
            vtM[vti] += M
    print_progress(1)
    print()

    # Normalize columns of the matrix
    for M in vtM:
        row_sq_sum = numpy.sqrt((M * M).sum(axis = 0))
        row_sq_sum[row_sq_sum < 1e-10] = 1
        M /= row_sq_sum

    return vtM


def compute_bounding_box_barycentric_coords(fvts, height, width):
    """
    Computes the barycentric coordingate of pixels inside the bounding box.
    Returns the bounding box grid and the barycentric coordinates of the gird.
    """
    # Compute the bounding box of the face.
    xs = (width * fvts[:, 0]).T # Array of x values of the triangle
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
    coords = numpy.linalg.solve(B, grid)

    return grid, coords


def tangent_to_global_normals(mesh, texture):
    """
    Convert a tangent space normal map to a global space normal map.
    Input:
        mesh - a mesh in OBJ format
        texture - the tangent space texture map
    Output:
        Returns a global space normal map for the mesh.
    """
    vtM = compute_vt_transforms(mesh)
    vts = numpy.array(mesh.vt)

    height, width = texture.shape[:2]
    out_tex = numpy.zeros((width, height, 3))
    mask = numpy.zeros((height, width), dtype=int)
    for i, face in enumerate(mesh.f):
        print_progress(i / float(len(mesh.f)))

        fvts = numpy.vstack([vts[fi.vt] for fi in face])
        try:
            grid, coords = compute_bounding_box_barycentric_coords(fvts,
                height, width)
        except:
            continue

        # Get the transformation values for the corners as a 3D array
        corner_vals = numpy.array([vtM[fi.vt] for fi in face])
        # Interpolate the corner matricies and normalize columns
        interp_vals = corner_vals.T.dot(coords)
        interp_vals /= numpy.sqrt((interp_vals**2).sum(axis = 0))
        pixel_vals = texture[grid[1], grid[0], :3]

        are_inside = numpy.all(coords >= -1e-4, axis = 0)
        # pdb.set_trace()

        # Transform each pixel in the bounding box
        for j, is_inside in enumerate(are_inside):
            x, y = grid[:2, j]

            interp_M = interp_vals[:, :, j]

            # Global normal for the pixel value
            gNorm = interp_M.T.dot(texture[y, x, :3])
            # try:
            #     # Inverse transpose of the texture
            #     gNorm = numpy.linalg.solve(interp_M, texture[y, x, :3])
            # except:
            #     continue

            # Stored the transformed normal
            if is_inside:
                out_tex[y, x] = gNorm
                mask[y, x] = 2
            elif mask[y, x] == 1:
                out_tex[y, x] += gNorm
                out_tex[y, x] /= 2.0
            elif mask[y, x] == 0:
                out_tex[y, x] = gNorm
                mask[y, x] = 1

    # Normalize the pixels stored in the texture.
    row_sq_sum = numpy.sqrt((out_tex**2).sum(axis = 2))
    row_sq_sum[row_sq_sum < 1e-10] = 1
    out_tex /= row_sq_sum.reshape(height, width, 1)

    print_progress(1)
    print()
    return out_tex


def create_parser():
    """ Creates an ArgumentParser for this command line tool. """
    parser = argparse.ArgumentParser(description = "Convert a tangent normal" +
        " map to an object space normal map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s path/to/input_model path/to/input_texture [-h] " +
        "[-o path/to/output_texture] [-b]")
    parser.add_argument("in_mesh", metavar="path/to/input_model",
        help="Path to input mesh file.")
    parser.add_argument("in_texture", metavar="path/to/input_texture",
        help="Path to input texture image or directory to load all textures " +
        "from.")
    parser.add_argument("-o", "--output", metavar="path/to/output_texture",
        help="Name of output texture or directory to save batch textures.",
        dest="out_texture")
    parser.add_argument("-b", "--blender", action="store_true",
        dest="do_blender", help="Should the output be in blender mode?")
    return parser


def parse_args(parser=None):
    """
    Uses ArgumentParser to parse the command line arguments.
    Input:
        parser - a precreated parser (If parser is None, creates a new parser)
    Outputs:
        Returns the arguments as a tuple in the following order:
            (in_mesh, in_texture, out_texture, loadFromDirectory, loadFromData,
             method, sv_method)
    """
    if(parser is None):
        parser = create_parser()
    args = parser.parse_args()

    # Check that in_mesh exists
    if(not os.path.exists(args.in_mesh)):
        parser.error("Path to input mesh does not exist.")
    # Check that in_texture exists
    if(not os.path.exists(args.in_texture)):
        parser.error("Path to input texture(s) does not exist.")

    if(args.out_texture is None):
        # Split the input texture name by the extension.
        in_path, in_ext = os.path.splitext(args.in_texture)
        # Create a temporary output texture filename.
        if(args.do_blender):
            args.out_texture = in_path + '-object' + in_ext
        else:
            args.out_texture = in_path + '-global' + in_ext
    else:
        if(os.path.isdir(args.out_texture)):
            parser.error("Input texture is a file, but output is a directory.")

    return (args.in_mesh, args.in_texture, args.out_texture, args.do_blender)

if __name__ == "__main__":
    (in_mesh, in_texture, out_texture, do_blender) = parse_args()

    # Time the amount of time this takes.
    startTime = time.time()

    mesh = obj_reader.quads_to_triangles(obj_reader.load_obj(in_mesh))

    data = numpy.array(texture.load_texture(in_texture))
    isFloatTexture = not issubclass(data.dtype.type, numpy.integer)
    if(not isFloatTexture):
        data = data / 255.0
    data = data * 2 - 1

    converted_tex = tangent_to_global_normals(mesh, data)
    # pdb.set_trace()
    converted_tex = -converted_tex
    if(do_blender):
        converted_tex[:, :, 0] *= -1
    converted_tex = (converted_tex + 1) / 2.0
    # texture.save_texture_channels(out_texture, "./cow-smooth.tiff")
    converted_tex = to_uint8(converted_tex)
    texture.save_texture(converted_tex, out_texture)

    print("Total Runtime: %.2f seconds" % (time.time() - startTime))
