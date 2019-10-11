"""
Compute the Least Squares Constraints for the seams.

Written by Zachary Ferguson
"""

import scipy.sparse
import scipy.sparse.linalg

from .util import QuadEnergy


def constrain_values(mask, textureVec):
    """
        Constrain the values inside the triangles
        Inputs:
            mask - a height by width matrix of boolean values for if
                pixel r,c is to be constrained
            textureVec - a vector for the texture pixel values
        Output:
            A least square constrained matrices in the form (C, d)
    """
    height, width = mask.shape
    depth = (textureVec.shape + (1,))[1]

    C = scipy.sparse.diags(mask.ravel().astype(float)).tocsc()
    d = -scipy.sparse.csc_matrix(C.dot(textureVec))

    norm = mask.sum()
    if norm != 0:
        norm = 1.0 / norm

    # The matrix should be C.T*C, but C is a diagonal binary matrix so it
    # doesn't matter.
    # return QuadEnergy(norm*C.T.dot(C), norm*d, norm*d.T.dot(d))
    return QuadEnergy(norm * C, norm * d, norm * d.T.dot(d))
