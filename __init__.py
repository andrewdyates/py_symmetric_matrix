#!/usr/bin/python
"""Efficient representation of a symmetric matrix using pylab.

Also see: 
  scipy.spatial.distance.squareform
"""
import numpy



def sym_idx(x, y, n, with_diagonal=False):
  """Return symmetric array index.

  Args:
    x: int of row
    y: int of column
    n: int of square array size
    with_diagonal: bool if to consider diagonal entries
  Returns:
    int of array index in symmetric matrix implementation
  """
  assert all((x<n, y<n, x>=0, y>=0))
  # x is always <= y
  if x > y:
    x, y = y, x
  # Formula derivation on page 33 of lab notebook 2
  if not with_diagonal:
    assert x != y
    n -= 1
    y -= 1
  return (2*n-x-1)*x/2 + y





class SymmetricMatrix(object):
  """Efficient symmetric matrix stored as a numpy array.
  """
  def __init__(self, n=None, store_diagonal=True, dtype=numpy.float):
    """Initialize matrix.

    Args:
      n: int of matrix dimension
      store_diagonal: bool if to store the matrix diagonal
      dtype: obj of numpy datatype of matrix
    """
    assert n is not None
    self.n = n
    self.dtype = dtype
    self.store_diagonal = store_diagonal
    
    self.n_entries = (self.n) * (self.n+1) / 2
    if not store_diagonal:
      self.n_entries -= self.n
      
    self._m = numpy.zeros(self.n_entries, dtype=dtype)

  def get(self, x, y, _idx=None):
    if _idx is None:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    return self._m[idx]

  def set(self, x, y, value, _idx=None):
    if _idx is None:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    self._m[idx] = value
