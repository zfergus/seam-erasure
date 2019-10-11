"""
Script for loading generic texture data into a numpy array.
Written by Zachary Ferguson
"""

import os
import logging

import math

from PIL import Image


def load_texture(fname):
    """ Load a PIL Image with name fname. """
    logging.info("Loading Texture: %s" % os.path.abspath(fname))
    return Image.open(fname).transpose(Image.FLIP_TOP_BOTTOM)


def save_texture(data, fname):
    """ Save a PIL Image with name fname. """
    logging.info("Saving Texture: %s" % os.path.abspath(fname))
    texture = Image.fromarray(data).transpose(Image.FLIP_TOP_BOTTOM)
    texture.save(fname)


def save_texture_channels(data, base_fname):
    """ Save a N depth image as N images of depth 1. """
    data.reshape((data.shape + (1, ))[:3])
    depth = data.shape[2]
    base, ext = os.path.splitext(base_fname)
    fstr = "%%s-%%0%dd%%s" % int(math.log10(data.shape[2]) + 1)
    for i in range(depth):
        fname = fstr % (base, i, ext)
        save_texture(data[:, :, i], fname)


def save_float_mat_as_boolean(M, fname, tolerance=1e-8):
    """ Save a floating point matrix as a binary image for > tolerance. """
    assert len(M.shape) == 2  # Needs to be a 2-dimensional matrix
    tmp = 255 * (abs(M.A) > tolerance).astype("uint8")
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1).repeat(3, axis=2)
    # TODO: Use optional mode parameter to set to L of 1
    Image.fromarray(tmp).save(fname)


# Quick test of save all depths.
if __name__ == "__main__":
    import numpy
    img = numpy.array(load_texture("../img/rgb.png"))
    img = img / 255.0
    save_texture_channels(img, "output.tiff")
