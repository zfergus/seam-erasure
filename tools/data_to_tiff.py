#!/usr/bin/env python -u
"""
Load and convert a .data file into a series of #channel tiff images.
Written by Zachary Ferguson
"""

import os
import argparse

import numpy
from PIL import Image

import includes

import weight_data
import texture


def create_parser():
    """ Create an argument parser for parsing input and output paths. """
    parser = argparse.ArgumentParser(description="Convert a Data file to " +
        "single channel floating-point Tiff Images.",
        usage="%(prog)s path/to/input.data [-o path/to/output/]")
    parser.add_argument("in_path", metavar="path/to/input.data",
        help="Path to the input data file.")
    parser.add_argument("-o", "--output", metavar="path/to/output/",
        help="Name of the output directory to save tiff.", dest="out_path")
    return parser


def parse_args(parser=None):
    """ Uses ArgumentParser to parse the command line arguments. """
    if(parser is None):
        parser = create_parser()
    args = parser.parse_args()

    if(not os.path.exists(args.in_path)):
        parser.error("Path to input data does not exist.")
    if(not os.path.isfile(args.in_path)):
        parser.error("Input data is not a file.")

    if(args.out_path is None):
        args.out_path = os.path.dirname(args.in_path)

    return args

if __name__ == "__main__":
    # Get the arguments this file
    args = parse_args()

    # Get the input
    data = weight_data.read_tex_from_path(args.in_path)[0]

    # Save as tiffs
    if(not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    base, ext = os.path.splitext(os.path.basename(args.in_path))
    texture.save_texture_channels(data,
        os.path.join(args.out_path, base + ".tiff"))
