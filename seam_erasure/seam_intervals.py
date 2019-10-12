"""
Compute the intersection of seam edges with pixels of a width x height texture.

Written by Zachary Ferguson
"""

import math

from .util import UV_to_XY, range_min_max


def compute_edge_intervals(edge, width, height):
    """Compute the intervals along an edge."""
    interval = {0, 1}

    v0 = UV_to_XY(edge[0], width, height)
    v1 = UV_to_XY(edge[1], width, height)

    # Create expressions for converting to t values
    x1_x0 = float(v1.x - v0.x)

    def x_to_t(x):
        """Convert a x value to a paramertized t value allong the edge."""
        return round((x - v0.x) / (x1_x0), 10)

    y1_y0 = float(v1.y - v0.y)

    def y_to_t(y):
        """Convert a y value to a paramertized t value allong the edge."""
        return round((y - v0.y) / (y1_y0), 10)

    # Add whole number pixels to t values
    interval |= set(
        x_to_t(x) for x in range_min_max(math.ceil(v0.x), math.ceil(v1.x)))

    interval |= set(
        y_to_t(y) for y in range_min_max(math.ceil(v0.y), math.ceil(v1.y)))

    return interval


def compute_edgePair_intervals(edgePair, width, height):
    intervals = set()
    for edge in edgePair:
        intervals |= compute_edge_intervals(edge, width, height)
    return sorted(list(intervals))


def compute_seam_intervals(uv_seam, width, height):
    """Computes all intervals from 0 to 1 on the seam's edge pairs."""
    return [
        compute_edgePair_intervals(edgePair, width, height)
        for edgePair in uv_seam
    ]
