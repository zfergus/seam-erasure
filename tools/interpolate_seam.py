#!/usr/bin/env python -u
"""
Interpolate the values along the seam, saving them out to a image file.
Author: Zachary Ferguson
"""

from __future__ import print_function

import os
import sys
import argparse

import includes

from find_seam import find_seam, seam_to_UV
from seam_intervals import compute_edge_intervals
import obj_reader
import texture
from util import *


def create_parser():
    """ Create an argument parser for parsing input and output paths. """
    parser = argparse.ArgumentParser(description="Interpolate the values " +
        "along the seam of a 3D model.",
        usage=("%(prog)s path/to/input_model path/to/input_texture " +
        "[-o path/to/output_texture]"))
    parser.add_argument("in_mesh", metavar="path/to/input_model",
        help="Path to input mesh file.")
    parser.add_argument("in_texture", metavar="path/to/input_texture",
        help=("Path to input texture image or directory to load all " +
        "textures from."))
    parser.add_argument("-o", "--output", metavar="path/to/output_image",
        help="Name of output image to save interpolated seam.",
        dest="out_path")
    return parser


def parse_args(parser=None):
    if(parser is None):
        parser = create_parser()
    args = parser.parse_args()

    # Check that in_mesh exists
    if(not os.path.exists(args.in_mesh)):
        parser.error("Path to input mesh does not exist.")
    # Check that in_texture exists
    if(not os.path.exists(args.in_texture)):
        parser.error("Path to input texture(s) does not exist.")

    if(args.out_path is None):
        # Split the input texture name by the extension.
        in_path, in_ext = os.path.splitext(args.in_texture)
        # Create a temporary output texture filename.
        args.out_path = in_path + '-seam' + in_ext
        # Is the texture being loaded from a data file?
    else:
        if(os.path.isdir(args.out_path)):
            parser.error("Input texture is a file, but output is a " +
                "directory.")

    return (args.in_mesh, args.in_texture, args.out_path)


def interpolate_seam(mesh, textureData):
    """
    Solve for the seam values.
    Inputs:
        mesh - mesh object (probably should be an OBJ object)
        textureData - numpy array of the pixel values for the texture
    Output:
        Returns an array of pixel values for the interpolated seam.
    """

    print("\nFinding Seam:")

    height, width, depth = (textureData.shape + (1,))[:3]
    textureData = textureData.reshape(-1, depth)

    uv_seam, uv_boundary, uv_foldovers = seam_to_UV(mesh, *find_seam(mesh))

    print("Done\n\nComputing seam values:")

    edge_count = float(len(uv_seam) * 2)
    count = 0.0

    vals = []
    for edgePair in uv_seam:
        pair_vals = []
        for i, edge in enumerate(edgePair):
            # Print the progress of interpolating the seams.
            # print("%.2f%%" % (100 * count / edge_count))
            print_progress(count / edge_count)
            count += 1.0

            pair_vals.append([])
            edge_vals = pair_vals[i]

            # Convert from UV to XY values
            xy0 = UV_to_XY(edge[0], width, height)
            xy1 = UV_to_XY(edge[1], width, height)
            interval = sorted(list(
                compute_edge_intervals(edge, 2 * width, 2 * height)))

            # Iterate over the intervals
            for a, b in pairwise(interval):
                uv_mid = lerp_UV((a + b) / 2.0, edge[0], edge[1])
                pixelIs = surrounding_pixels(uv_mid, width, height,
                    as_index = True)
                p00, p01, p10, p11 = [textureData[i] for i in pixelIs]
                u, v = globalUV_to_local(uv_mid, pixelIs[0] % width,
                    pixelIs[0] // width, width, height)
                val = (1 - v) * ((1 - u) * p00 + u * p01) + v * (
                    (1 - u) * p10 + u * p11)
                edge_vals.append(val)
        vals.append(pair_vals)
    print_progress(1.0)
    print("\n")

    seamData = numpy.zeros((2 * width, 2 * height, depth))

    for i, pair_vals in enumerate(vals):
        for j, edge_vals in enumerate(pair_vals):
            v0, v1 = [XY(2 * width * uv.u, 2 * height * uv.v)
                for uv in uv_seam[i][j]]
            for k, val in enumerate(edge_vals):
                t = k / float(len(edge_vals))
                xy = lerp_XY(t, v0, v1)
                seamData[int(xy.y), int(xy.x)] = val

    return seamData

if __name__ == "__main__":
    in_mesh, in_texture, out_path = parse_args()

    mesh = obj_reader.quads_to_triangles(obj_reader.load_obj(in_mesh))

    textureData = numpy.array(texture.load_texture(in_texture))
    isFloatTexture = not issubclass(textureData.dtype.type, numpy.integer)
    if(not isFloatTexture):
        textureData = textureData / 255.0

    interpolated_seam = interpolate_seam(mesh, textureData)

    # Save the solved texture
    if(interpolated_seam.shape[2] < 2):
        interpolated_seam = numpy.squeeze(interpolated_seam, axis=2)

    if(not isFloatTexture):
        interpolated_seam = to_uint8(interpolated_seam)
    texture.save_texture(interpolated_seam, out_path)
