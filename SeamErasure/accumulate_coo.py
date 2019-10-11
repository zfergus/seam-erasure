"""
Faster way of accumulating sparse COO matricies. Avoids expensive loop of adds.

Written by Yotam Gingold
"""
import numpy
import scipy.sparse


class AccumulateCOO(object):
    """
    Class for accumulating additions of COO matricies. Does not sum matrices
    until total() is called.
    """

    def __init__(self):
        """Create an empty COO matrix accumulator."""
        self.row = []
        self.col = []
        self.data = []

    def add(self, A):
        """
        Add a coo_matrix to this matrix. Does not perform the addition until
        total() is called.

        Input:
            A - A coo_matrix to add to this matrix
        """
        self.row.append(A.row)
        self.col.append(A.col)
        self.data.append(A.data)

    def total(self, shape):
        """
        Constructs a coo_matrix from the accumulated values.

        Input:
            shape - shape of the output matrix
        Output:
            Return a coo_matrix of the accumulated values.
        """
        assert len(self.row) == len(self.col)
        assert len(self.row) == len(self.data)

        return scipy.sparse.coo_matrix((numpy.concatenate(self.data), (
            numpy.concatenate(self.row), numpy.concatenate(self.col))),
            shape=shape)
