import sys

import numpy
from PIL import Image

import context

from SeamErasure.texture import load_texture

assert(len(sys.argv) == 3)

image_a = numpy.array(load_texture(sys.argv[1]), dtype=float)
image_b = numpy.array(load_texture(sys.argv[2]), dtype=float)

assert(image_a.shape == image_b.shape)
assert(numpy.linalg.norm(image_a - image_b) < 1 / 255.)
