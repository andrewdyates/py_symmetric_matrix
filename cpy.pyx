#!/usr/bin/python
from __future__ import division
"""Efficient representation of a symmetric matrix using pylab.

Also see: 
  scipy.spatial.distance.squareform
"""
import numpy as np
cimport numpy as np

cdef extern from "math.h": 
  float sqrtf(float) 
  float abs(float) 
  float expf(float)
  float ceil(float) 


cpdef int sym_idx(int x, int y, int n, char with_diagonal=False):
  """Return symmetric array index.

  Args:
    x: int of row
    y: int of column
    n: int of square array size
    with_diagonal: bool if to consider diagonal entries
  Returns:
    int of array index in symmetric matrix implementation
  """
    # x is always <= y
  if x > y:
    x, y = y, x
  # Formula derivation on page 33 of lab notebook 2
  if not with_diagonal:
    n -= 1
    y -= 1
  return (2*n-x-1)*x//2 + y

cpdef inv_sym_idx(int idx, int n, char with_diagonal=False):
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
  cdef int c = 2*idx - n*(n+1)
  # quadradic equation to solve for q
  cdef float i_f = (-1 + sqrtf(1-4*c))/2
  cdef int i = (<int>ceil(i_f))
  
  x = n-i
  y = idx - (n*(n+1)//2 - i*(i+1)//2) + x
  if not with_diagonal:
    y += 1
  return x,y


cdef class SymmetricMatrix:
  """Efficient symmetric matrix stored as a np array.
  Initializes all values to zero.

  Note: this is not dramatically more efficient because the type of np.matrix
  is unknown. Optimizing as a 'cdef' is about a 30% improvement.
  """
  cdef public int n
  cdef public char store_diagonal
  cdef public dtype
  cdef public int n_entries
  cdef public np.ndarray _m
  def __init__(self, int n, char store_diagonal=True, np.ndarray matrix=None, dtype=np.float):
    """Initialize matrix.

    Args:
      n: int of matrix dimension
      store_diagonal: bool if to store the matrix diagonal
      matrix: object of np.array to use rather than new zero matrix
      dtype: obj of np datatype of matrix
    """
    self.n = n
    self.dtype = dtype
    self.store_diagonal = store_diagonal
    
    self.n_entries = (self.n) * (self.n+1) // 2
    if not store_diagonal:
      self.n_entries -= self.n

    if matrix is None:
      self._m = np.zeros(self.n_entries, dtype=dtype)
    else:
      self._m = matrix

  def get(self, int x=-1, int y=-1, int _idx=-1):
    if _idx == -1:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    else:
      assert x == -1 and y == -1
      idx = _idx
    return self._m[idx]

  def set(self, int x=-1, int y=-1, value=None, int _idx=-1):
    if value is None:
      value = 1
    if _idx is -1:
      idx = sym_idx(x, y, self.n, self.store_diagonal)
    else:
      assert x == -1 and y == -1
      idx = _idx
    self._m[idx] = value


class NamedSymmetricMatrix(SymmetricMatrix):
  """SymmetricMatrix with named rows and columns."""
  
  def __init__(self, var_list=None, **kwds):
    assert var_list is not None
    self.vars = dict([(name, idx) for idx, name in enumerate(var_list)])
    super(NamedSymmetricMatrix, self).__init__(n=len(var_list), **kwds)
    
  def get(self, x=None, y=None, _idx=None):
    assert bool(_idx is None) != bool(x is None and y is None)
    if _idx is None:
      assert self.store_diagonal or x != y
      i, j = self.vars[x], self.vars[y]
      return super(NamedSymmetricMatrix, self).get(x=i, y=j)
    else:
      return super(NamedSymmetricMatrix, self).get(_idx=_idx)

  def get_idx(self, x, y):
    assert self.store_diagonal or x != y
    i, j = self.vars[x], self.vars[y]
#    return sym_idx(i, j, self.n, with_diagonal=self.store_diagonal)
    return sym_idx(i, j, self.n, self.store_diagonal)

  def set(self, x=None, y=None, value=None, _idx=None):
    assert bool(_idx is None) != bool(x is None and y is None)
    if _idx is None:
      assert self.store_diagonal or x != y
      i, j = self.vars[x], self.vars[y]
      super(NamedSymmetricMatrix, self).set(x=i, y=j, value=value)
    else:
      super(NamedSymmetricMatrix, self).set(value=value, _idx=_idx)
      
  
