#!/usr/bin/python
from __init__ import *
import unittest

DIAG = [(0,0,0), (0,1,1), (0,2,2), (0,3,3), (1,1,4), (1,2,5), (1,3,6), (2,2,7), (2,3,8), (3,3,9)]
NO_DIAG = [(0,1,0), (0,2,1), (0,3,2), (1,2,3), (1,3,4), (2,3,5)]

class TestKnownBadIndex(unittest.TestCase):
  def test_known_bad_idx(self):
    n = 22184
    x,y= inv_sym_idx(22164, n)
    self.assertEqual(x, 0)
    self.assertEqual(y, 22165)
    x,y= inv_sym_idx(22165, n)
    self.assertEqual(x, 0)
    self.assertEqual(y, 22166)


class TestIdx(unittest.TestCase):

  def test_named_withdiag(self):
    varlist=['a', 'b', 'c']
    N = NamedSymmetricMatrix(var_list=varlist, store_diagonal=True)
    self.assertEqual(N.get_idx('a', 'b'), N.get_idx('b', 'a'))
    N.set('a', 'b', value=2)
    self.assertEqual(N.get('a','b'), 2)
    self.assertEqual(N.get('b','a'), N.get('a','b'))
    i = N.get_idx('a', 'b')
    N.set(_idx=i, value=3)
    self.assertEqual(N.get('a','b'), 3)
    self.assertEqual(N.get('b','a'), N.get('a','b'))
    N.set(_idx=i+1, value=5)
    self.assertEqual(N.get('a','b'), 3)
    self.assertEqual(N.get('b','a'), N.get('a','b'))

  def test_named_nodiag(self):
    varlist=['a', 'b', 'c']
    N = NamedSymmetricMatrix(var_list=varlist, store_diagonal=False)
    self.assertEqual(N.get_idx('a', 'b'), N.get_idx('b', 'a'))
    N.set('a', 'b', value=2)
    self.assertEqual(N.get('a','b'), 2)
    self.assertEqual(N.get('b','a'), N.get('a','b'))
    i = N.get_idx('a', 'b')
    N.set(_idx=i, value=3)
    self.assertEqual(N.get('a','b'), 3)
    self.assertEqual(N.get('b','a'), N.get('a','b'))
    N.set(_idx=i+1, value=5)
    self.assertEqual(N.get('a','b'), 3)
    self.assertEqual(N.get('b','a'), N.get('a','b'))
    
  
  def test_matrix(self):
    a = [1,2,3,4]
    m = SymmetricMatrix(n=4, dtype=int)
    self.assertEqual(m.n_entries, 10)
    
    for i in range(4):
      for j in range(4):
        m.set(i,j,i+j)

    for i in range(4):
      for j in range(4):
        self.assertEqual(m.get(i,j), i+j)

  def test_matrix_no_diag(self):
    a = [1,2,3,4]
    m = SymmetricMatrix(n=4, dtype=int, store_diagonal=False)
    self.assertEqual(m.n_entries, 6)
    
    for i in range(4):
      for j in range(4):
        if i == j: continue
        m.set(i,j,i+j)
    for i in range(4):
      for j in range(4):
        if i == j: continue
        self.assertEqual(m.get(i,j), i+j)
        
  def test_idx(self):
    """Verify index values for 4x4 matrix with diagonal."""
    n = 4
    for x,y,i in DIAG:
      self.assertEqual(sym_idx(x,y,n, with_diagonal=True), i)
      
  def test_idx_no_diagonal(self):
    """Verify index values for 4x4 matrix without diagonal."""
    n = 4
    for x,y,i in NO_DIAG:
      self.assertEqual(sym_idx(x,y,n,with_diagonal=False), i)

  def test_inv_idx_no_diagonal(self):
    """Verify inverse index values for 4x4 matrix without diagonal."""
    n = 4
    for x,y,i in NO_DIAG:
      self.assertEqual(inv_sym_idx(i,n,with_diagonal=False), (x, y))

  def test_inv_idx_diagonal(self):
    """Verify inverse index values for 4x4 matrix with diagonal."""
    n = 4
    for x,y,i in DIAG:
      self.assertEqual(inv_sym_idx(i,n,with_diagonal=True), (x, y))


if __name__ == "__main__":
  unittest.main()
