"""
Utility file for testing if points are in a given triangle.

Written by Zachary Ferguson
"""

import numpy


def points_in_triangle(tri, points, tol=1e-8):
    """
    Test if the points are inside the triangle.

    Input:
        tri - the triangle as a matrix where the rows are the xy points.
        points - the points as a matrix where the rows are the xy points.
    Returns a vector of boolean values.
    """
    # B is the transformation from xy to barycentric coordinates
    B = numpy.vstack([tri.T, numpy.ones(3)])

    vecs = numpy.vstack([points.T, numpy.ones((1, points.shape[0]))])

    # Convert the grid from XY locations to barycentric coordinates.
    # This will only fail of the triangle is degenerate.
    try:
        coords = numpy.linalg.solve(B, vecs)
    except Exception:
        return numpy.zeros(points.shape[0]).astype(bool)

    return numpy.all(coords >= -tol, axis=0)
