#!/usr/bin/python
from __future__ import division
"""Efficient representation of a symmetric matrix using pylab.

Also see: 
  scipy.spatial.distance.squareform
"""
import numpy as np


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
  assert all((x<n, y<n, x>=0, y>=0)), "x:%d, y:%d, n:%d" % (x, y, n)
  # x is always <= y
  if x > y:
    x, y = y, x
  # Formula derivation on page 33 of lab notebook 2
  if not with_diagonal:
    assert x != y
    n -= 1
    y -= 1
  return (2*n-x-1)*x//2 + y

def inv_sym_idx(idx, n, with_diagonal=False):
  """Return (x,y) given linear index to symmetric matrix (invert sym_idx).

  Args:
    idx: int of index to invert
    n: int size of matrix (number of variables)
    with_diagonal: include diagonal in matrix
  Returns:
    (x,y) of 0-indexed variable index.
  """
  if not with_diagonal:
    n -= 1
  c = 2*idx - n*(n+1)
  assert c <= 0
  # quadradic equation to solve for q
  i = (-1 + np.sqrt(1-4*c))/2
  i = int(np.ceil(i))
  
  x = n-i
  y = idx - (n*(n+1)//2 - i*(i+1)//2) + x
  if not with_diagonal:
    y += 1
  return x,y
  



class SymmetricMatrix(object):
  """Efficient symmetric matrix stored as a np array.
  Initializes all values to zero.
  """
  def __init__(self, n=None, store_diagonal=True, matrix=None, dtype=np.float):
    """Initialize matrix.

    Args:
      n: int of matrix dimension
      store_diagonal: bool if to store the matrix diagonal
      matrix: object of np.array to use rather than new zero matrix
      dtype: obj of np datatype of matrix
    """
    assert n is not None
    self.n = n
    self.dtype = dtype
    self.store_diagonal = store_diagonal
    
    self.n_entries = (self.n) * (self.n+1) / 2
    if not store_diagonal:
      self.n_entries -= self.n

    if matrix is None:
      self._m = np.zeros(self.n_entries, dtype=dtype)
    else:
      self._m = matrix

  def get(self, x=None, y=None, _idx=None):
    if _idx is None:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    else:
      assert x is None and y is None
      idx = _idx
    return self._m[idx]

  def set(self, x=None, y=None, value=None, _idx=None):
    if value is None:
      value = 1
    if _idx is None:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    else:
      assert x is None and y is None
      idx = _idx
    self._m[idx] = value


class NamedSymmetricMatrix(SymmetricMatrix):
  """SymmetricMatrix with named rows and columns."""
  
  def __init__(self, var_list=None, **kwds):
    assert var_list is not None
    self.vars = dict([(name, idx) for idx, name in enumerate(var_list)])
    super(NamedSymmetricMatrix, self).__init__(n=len(var_list), **kwds)
    
  def get(self, x, y, _idx=None):
    i, j = self.vars[x], self.vars[y]
    return super(NamedSymmetricMatrix, self).get(i, j, _idx=_idx)

  def get_idx(self, x, y):
    i, j = self.vars[x], self.vars[y]
    return inv_sym_idx(i, j, self.n)

  def set(self, x, y, value, _idx=None):
    i, j = self.vars[x], self.vars[y]
    super(NamedSymmetricMatrix, self).set(i, j, value, _idx=_idx)
  
