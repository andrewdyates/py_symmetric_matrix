#!/usr/bin/python
from __init__ import *
import unittest

DIAG = [(0,0,0), (0,1,1), (0,2,2), (0,3,3), (1,1,4), (1,2,5), (1,3,6), (2,2,7), (2,3,8), (3,3,9)]
NO_DIAG = [(0,1,0), (0,2,1), (0,3,2), (1,2,3), (1,3,4), (2,3,5)]

class TestIdx(unittest.TestCase):
  
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
