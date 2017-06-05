"""
Convert an RGB image to object normal for Blender. This done by inverting the
GB channels.

Written by Zachary Ferguson
"""

import numpy

import includes

import texture
import util


def convert_norms_for_blender(tex):
    """ Invert the green and blue channels. """
    bg = tex.sum(axis=2) == 0

    tex = 2 * tex - 1

    tex[:, :, 1:3] *= -1
    tex = (tex + 1) * 0.5

    tex[bg] = 0
    return tex

if __name__ == "__main__":
    in_name = (
        "/home/zachary/Downloads/HB_bronze_norm-straighten-0.01-Global.tiff")
    out_name = "/home/zachary/Downloads/out.tiff"

    tex = numpy.array(texture.load_texture(in_name))

    out_tex = convert_norms_for_blender(numpy.copy(tex / 255.))

    out_tex = util.to_uint8(out_tex)
    texture.save_texture(out_tex, out_name)
