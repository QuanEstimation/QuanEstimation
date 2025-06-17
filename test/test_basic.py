import pytest
import numpy as np
from quanestimation.Common.Common import basis, gramschmidt, suN_generator

def test_basis():
    result = basis(2, 0)
    assert np.array_equal(result, np.array([[1], [0]])) == 1

def test_gramschmidt():
    A = np.array([[1, 0], 
                  [1, 1]], dtype=np.complex128)
    result = gramschmidt(A)
    assert np.array_equal(result, np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])) == 1

def test_suN_generator():
    n = 2
    result = suN_generator(n)
    expected = (np.array([[1, 0], [0, -1]]), 
                np.array([[0, 1], [1, 0]]), 
                np.array([[0, 0], [0, 1]]))
    assert (np.array_equal(result[0], expected[0]) and 
            np.array_equal(result[1], expected[1]) and 
            np.array_equal(result[2], expected[2])) == 1    