"""Compare the value of two images."""

import sys

import numpy

import context  # noqa

from seam_erasure.texture import load_texture

assert(len(sys.argv) == 3)

image_a = numpy.array(load_texture(sys.argv[1]), dtype=float) / 255.
image_b = numpy.array(load_texture(sys.argv[2]), dtype=float) / 255.

assert(image_a.shape == image_b.shape)
assert(numpy.linalg.norm(image_a - image_b) < 1 / 255.)
