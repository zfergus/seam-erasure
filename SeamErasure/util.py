"""
Collection of utility functions for wrapping-textures.

Written by Zachary Ferguson
"""

from __future__ import print_function

import sys
import time
import itertools
import logging

import numpy
from recordclass import recordclass

######################################
# Record classes for neccessary data #
######################################
UV = recordclass('UV', ['u', 'v'])
Pixel = recordclass('Pixel', ['x', 'y'])
XY = recordclass('XY', ['x', 'y'])
XYZ = recordclass('XYZ', ['x', 'y', 'z'])

# Quadtratic energy: x.T @ Q @ x + 2 * x.T @ L + C = 0
QuadEnergy = recordclass('QuadraticEnergy', ['Q', 'L', 'C'])


def pairwise(iterable):
    """Returns: s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def pairwise_loop(iterable):
    """
    Create pair wise list of the iterable given with the last element being the
    first.

    Returns: s -> (s0,s1), (s1,s2), (s2, s3), ..., (sN, s0)
    """
    return tuple(pairwise(iterable)) + ((iterable[-1], iterable[0]), )


def rowcol_to_index(row, col, width):
    """Convert row major coordinates to 1-D index."""
    return width * row + col


def lerp(t, x0, x1):
    """Linearly interpolate between x0 and x1."""
    return x0 + t * (x1 - x0)


def lerpPair(t, p0, p1):
    """Linearly interpolate independent indexed paires."""
    return [lerp(t, p0[0], p1[0]), lerp(t, p0[1], p1[1])]


def lerp_UV(t, uv0, uv1):
    """
    Linearly interpolate between (u0,v0) and (u1,v1).

    Returns a UV object.
    """
    return UV(*lerpPair(t, uv0, uv1))


def lerp_XY(t, xy0, xy1):
    """
    Linearly interpolate between (x0,y0) and (x1,y1).

    Returns a XY object.
    """
    return XY(*lerpPair(t, xy0, xy1))


def UV_to_XY(uv, width, height, is_clamped=False):
    """
    Convert the given UV to XY coordinates.

    uv is defined in terms of GPU UV space.
    """
    # s*width - 0.5; t*height - 0.5
    xy = XY(x=uv.u * width - 0.5, y=uv.v * height - 0.5)

    if is_clamped:
        xy = (
            numpy.clip(xy[0], 0, max(0, width - 1)),
            numpy.clip(xy[1], 0, max(0, height - 1)))
    return xy


def UVs_to_XYs(uvEdges, width, height):
    """Convert a UV edge to XY space in the texture."""
    return [UV_to_XY(vert, width, height) for edge in uvEdges for vert in edge]


def globalUV_to_local(uv, minX, minY, width, height):
    """
    Convert from a texture's global UV to local UV.

    Local pixel values defined by the minimum x and y values.
    uv is defined in terms of GPU UV space.
    """
    x, y = UV_to_XY(uv, width, height, True)
    return UV(u=x - minX, v=y - minY)


def globalEdge_to_local(uv0, uv1, minI, width, height):
    """
    Convert a edge from a texture's global UV to local UV.

    Local pixel values defined by the minimum x and y values.
    uv is defined in terms of GPU UV space.
    """
    minX = minI % width
    minY = minI // width
    return [
        globalUV_to_local(uv, minX, minY, width, height) for uv in (uv0, uv1)
    ]


def surrounding_pixels(uv, w, h, as_index=False, as_tuple=False):
    """
    Determine the surrounding pixels of the given point at (u,v).

    uv is defined in terms of GPU UV space.
    Returns a Tuple of surrounding four Pixel objects.
    Pixels are ordered as: (Lower Left, Lower Right, Upper Left, Upper Right)
    """
    assert not (as_index and as_tuple)

    # Convert from GPU UV coordinates to XY coordinates
    (x, y) = UV_to_XY(uv, w, h, is_clamped=True)

    # Convert from XY to Pixel coordinates
    px = int(min(max(0, numpy.floor(x)), w - 2))  # X in Range(0,w-1)
    py = int(min(max(0, numpy.floor(y)), h - 2))  # Y in Range(0,h-1)

    p00 = Pixel(x=px, y=py)

    px = int(min(max(0, numpy.floor(x) + 1), w - 1))  # X in Range(0,w-1)
    py = int(min(max(0, numpy.floor(y) + 1), h - 1))  # Y in Range(0,h-1)

    p11 = Pixel(x=px, y=py)

    # Create tuple of soronding pixels in Pixel Space
    ps = (p00, Pixel(x=p11.x, y=p00.y), Pixel(x=p00.x, y=p11.y), p11)

    # If requested, convert from Pixel space to 1D index space
    if as_index:
        return [rowcol_to_index(p.y, p.x, w) for p in ps]
    if as_tuple:
        return tuple(tuple(p) for p in ps)
    return ps


def range_min_max(a, b):
    """Create a range from the min value to the max value."""
    return range(int(min(a, b)), int(max(a, b)))


def print_dots(time_delta=1.0):
    """
    Print out a dot every time_delta seconds.

    Loop after three dots.
    """
    dot_count = 0
    while True:
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            dot_count = (dot_count % 3) + 1
            print(("." * dot_count) + (" " * 3), end="\r")
            sys.stdout.flush()
        time.sleep(time_delta)


def verts_equal(v0, v1, epsilon=1e-8):
    """
    Test if two given vertices are equal within a certain epsilon.

    WARNING: This is slower than ==, but it allows for a tolerance level of
    equality.
    """
    assert epsilon >= 0.0

    if len(v0) != len(v1):
        return False

    for a, b in zip(v0, v1):
        if (abs(a - b) > epsilon):
            return False
    return True


def normalize_array(arr):
    """Normalize the given array to be in range [0,1]."""
    minVal = numpy.amin(arr)
    maxVal = numpy.amax(arr)
    return (arr - minVal) / float(maxVal - minVal)


def is_counterclockwise(v0, v1, v2):
    """
    Determine if the triangle defined by the given vertices in
    counter-clockwise order.

    Input:
        v0, v1, v2 - 2D coordinates for the vertices of the triangle
    Output:
        Returns True if the triangle is counter-clockwise order.
    """
    mat = numpy.array([[1, v[0], v[1]] for v in (v0, v1, v2)])
    return numpy.linalg.det(mat) > 0


# Convert back to image format
def to_uint8(data, normalize=False):
    """Convert the data in a floating-point vector to unsigned bytes."""
    # Normilize the solved values.
    if (normalize):
        data = normalize_array(data)

    for i in range(data.shape[0]):
        data[i] = data[i].clip(0.0, 1.0)
    data = (data * 255).round().astype("uint8")
    return data


def save_ijvs(A, fname):
    """Save a sparse matrix as a list of ijv pairings."""
    A = A.tocoo()
    height, width = A.shape
    M = numpy.empty((A.row.shape[0], 3))
    M[:, 0] = A.row
    M[:, 1] = A.col
    M[:, 2] = A.data
    lines = ["%d %d %.17f\n" % (ijv[0], ijv[1], ijv[2]) for ijv in M]
    with open(fname, "w") as f:
        f.write("%d %d\n" % (height, width))
        for line in lines:
            f.write(line)


def save_dense(A, fname):
    """Save an array as a text file, one line per row."""
    m, n = A.shape
    with open(fname, "w") as f:
        for row in A:
            for val in row:
                f.write("%.17f " % val)
            f.write("\n")
