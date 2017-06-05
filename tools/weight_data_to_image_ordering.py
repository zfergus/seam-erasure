#!/usr/bin/env python
""" Convert data in OpenGL row ordering to Image ordering. """

from __future__ import print_function

import sys

import includes

import weight_data


def weight_data_to_image_ordering(data):
    """ Convert data in OpenGL row ordering to Image ordering. """
    return data[::-1]

if __name__ == "__main__":
    def usage():
        print("Usage:", sys.argv[0], "path/to/input.data [path/to/out.data]",
              file = sys.stderr)
        sys.exit(-1)

    if len(sys.argv) == 2:
        in_path = sys.argv[1]
        # Split the input texture name by the extension.
        start, ext = os.path.splitext(args.in_texture)
        # Create a temporary output texture filename.
        out_path = start + '-image-ordering' + ext
    elif len(sys.argv) == 3:
        in_path, out_path = sys.argv[1:]
    else:
        usage()

    weight_data.write_tex_to_path(out_path, weight_data_to_image_ordering(
        weight_data.read_tex_from_path(in_path)[0]))
