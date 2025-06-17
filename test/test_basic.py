import pytest
import numpy as np
from quanestimation.Common.Common import basis, gramschmidt

def test_basis():
    result = basis(2, 0)
    assert np.array_equal(result, np.array([[1], [0]])) == 1

def test_gramschmidt():
    A = np.array([[1, 0], 
                  [1, 1]], dtype=np.complex128)
    result = gramschmidt(A)
    assert np.array_equal(result, np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])) == 1