#!/usr/bin/env python
"""
CMD-line tool for running seam erasure.

Written by Zachary Ferguson
"""
from __future__ import print_function, division

import os
import argparse
import time
import logging

from recordclass import recordclass

import numpy

from . import obj_reader
from . import texture
from . import seam_erasure
from .util import to_uint8
from . import weight_data

InputTextureFile = recordclass(
    "InputTextureFile", ["name", "depth", "isFloat", "isDataFile"])


def create_parser():
    """Create an ArgumentParser for this command line tool."""
    parser = argparse.ArgumentParser(
        description=("Erase texture seams to prevent visible seams or tearing "
                     "in various texture maps (color, normal, displacement, "
                     "ambient occlusion, etc.)"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage=("%(prog)s path/to/input_model path/to/input_texture [-h] "
               "[-o path/to/output_texture] [-g] [--sv {none,texture,lerp}] "
               "[-d]"))
    parser.add_argument("in_mesh", metavar="path/to/input_model",
                        help="Path to input mesh file.")
    parser.add_argument("in_texture", metavar="path/to/input_texture",
                        help=("Path to input texture image or directory to "
                              "load all textures from."))
    parser.add_argument(
        "-o", "--output", metavar="path/to/output_texture", dest="out_texture",
        help="Name of output texture or directory to save batch textures.")
    parser.add_argument(
        "-g", "--global", action="store_true", dest="do_global",
        help="Should the minimization have global effects?")
    parser.add_argument(
        "--sv", choices=["none", "texture", "lerp"], default="none",
        dest="sv_method",
        help=("What method should be used to compute the seam value energy? "
              "None implies do not use seam value. "
              "Texture implies use difference in originial texture. "
              "Lerp implies use linearly interpolated values along the edge."))
    parser.add_argument(
        "-d", "--data", action="store_true", dest="loadFromData",
        help="Should the input texture(s) be loaded as data files?")
    return parser


def parse_args(parser=None):
    """
    Use ArgumentParser to parse the command line arguments.

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

    loadFromDirectory = os.path.isdir(args.in_texture)
    if(args.out_texture is None):
        if(loadFromDirectory):
            args.out_texture = os.path.normpath(args.in_texture +
                                                "/erased") + "/"
        else:
            # Split the input texture name by the extension.
            in_path, in_ext = os.path.splitext(args.in_texture)
            # Create a temporary output texture filename.
            args.out_texture = in_path + '-erased' + in_ext
    else:
        if(loadFromDirectory):
            if(os.path.isfile(args.out_texture)):
                parser.error("Input texture is a directory, but output is a " +
                             "file.")
            args.out_texture += "/"
        else:
            if(os.path.isdir(args.out_texture)):
                parser.error("Input texture is a file, but output is a " +
                             "directory.")

    sv_methods = {"none": seam_erasure.SeamValueMethod.NONE,
                  "texture": seam_erasure.SeamValueMethod.TEXTURE,
                  "lerp": seam_erasure.SeamValueMethod.LERP}
    sv_method = sv_methods[args.sv_method]

    if(loadFromDirectory and sv_method == sv_methods["lerp"]):
        parser.error("Unable to perform seam value energy computation using " +
                     "lerp while performing batch texture solving.")

    return (args.in_mesh, args.in_texture, args.out_texture, loadFromDirectory,
            args.loadFromData, sv_method, args.do_global)


def loadTextures(in_path, loadFromDirectory, loadFromData):
    """
    Load all textures into a single height x width x depth(s) array.

    Inputs:
        in_path - path to texture(s)
        loadFromDirectory - is the path a directory?
        loadFromData - should all textures be treated as data files?
    Output:
        Returns array of texture data and list of cooresponding
        InputTextureFile objects.
    """
    if(loadFromDirectory):
        files = sorted([f for f in os.listdir(in_path) if
                        os.path.isfile(os.path.join(in_path, f))])
    else:
        in_path, base = os.path.split(in_path)
        files = [base]

    textures = []
    textureData = None
    for f in files:
        fpath = os.path.join(in_path, f)
        if(loadFromData or os.path.splitext(f)[1] == ".data"):
            data = weight_data.read_tex_from_path(fpath)[0]
            isFloatTexture, isDataFile = True, True
        else:
            data = numpy.array(texture.load_texture(fpath))
            isFloatTexture = not issubclass(data.dtype.type, numpy.integer)
            if(not isFloatTexture):
                data = data / 255.0
            isDataFile = False

        data = numpy.expand_dims(data, -1) if len(data.shape) < 3 else data
        textures.append(
            InputTextureFile(f, data.shape[2], isFloatTexture, isDataFile))
        textureData = data if textureData is None else (
            numpy.concatenate((textureData, data), axis=2))

    return textureData, textures


def saveTextures(outData, textures, out_path, loadFromDirectory):
    """
    Save the textures to the specified path.

    Inputs:
        outData - height x width x depth array for the output texture(s)
        textures - list of InputTextureFile objects
        out_path - path to output
        loadFromDirectory - is the output path a directory?
    Output:
        Saves textures.
    """
    assert len(outData.shape) == 3

    out_dir = os.path.dirname(out_path)
    if(out_dir != "" and not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    current_depth = 0
    for textureFile in textures:
        next_depth = current_depth + textureFile.depth

        textureData = outData[:, :, current_depth:next_depth]
        if(textureData.shape[2] < 2):
            textureData = numpy.squeeze(textureData, axis=2)

        if(not textureFile.isFloat):
            textureData = to_uint8(textureData, normalize=False)

        # Save the solved texture
        if(loadFromDirectory):
            base, ext = os.path.splitext(textureFile.name)
            out_texture = os.path.join(out_path, base + "-erased" + ext)
        else:
            out_texture = out_path

        if(textureFile.isDataFile):
            weight_data.write_tex_to_path(out_texture, textureData)
        else:
            texture.save_texture(textureData, out_texture)

        current_depth = next_depth


def main():
    logging.basicConfig(
        # format="[%(levelname)s] [%(asctime)s] %(message)s",
        format="%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO)
    (in_mesh, in_texture, out_texture, loadFromDirectory, loadFromData,
        sv_method, do_global) = parse_args()

    # Time the amount of time this takes.
    startTime = time.time()

    mesh = obj_reader.quads_to_triangles(obj_reader.load_obj(in_mesh))

    textureData, textures = (
        loadTextures(in_texture, loadFromDirectory, loadFromData))

    height, width = textureData.shape[:2]

    logging.info("Model: %s\nTexture: %s\n" % (os.path.basename(in_mesh),
                                               os.path.basename(in_texture)))

    out = seam_erasure.erase_seam(
        mesh, textureData, sv_method=sv_method, do_global=do_global)

    minVal = out.min()
    maxVal = out.max()

    logging.info("Min/Max of solved values:\nMin: %g\nMax: %g\n" %
                 (minVal, maxVal))

    saveTextures(out.reshape((height, width, -1)), textures, out_texture,
                 loadFromDirectory)

    logging.info("\nTotal Runtime: %.2f seconds" % (time.time() - startTime))


if __name__ == "__main__":
    main()
