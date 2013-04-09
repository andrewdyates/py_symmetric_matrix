py_symmetric_matrix
===================

Efficient symmetric matrix using pylab in Python.
Upper triangle, column then row enumeration.
Supports efficient indexing by name.

Vector form (without diagonal) is the same used by:
scipy.spatial.distance.squareform
Reconstitute full matrix from symmetric vector v like:
M = squareform(v)


CYTHON EXTENSION
------------------------------
BUILD for cython:
$ python setup_cython.py build_ext --inplace

IMPORT for cython:
from py_symmetric_matrix.cpy import *

Note: SymmetricMatrix-based classes are not fully optimized because
the data type of the underlying numpy matrix is dynamic. Best cython numpy
performance is on statically typed numpy matrices.