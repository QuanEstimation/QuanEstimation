import pytest
from quanestimation.Resource.Resource import SpinSqueezing
import numpy as np

def test_SpinSqueezing():
    
    # Collective spin operators
    Jy = np.array(
            [[0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.-1.22474487j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+1.22474487j, 0.+0.j, 0.-1.22474487j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+1.22474487j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j]]
             )
    Jz = np.array(
            [[2.,  0.,  0.,  0., 0.],
             [0.,  1.,  0.,  0., 0.],
             [0.,  0.,  0.,  0., 0.],
             [0.,  0.,  0.,  -1., 0.],
             [0.,  0.,  0.,  0., -2.]]
             )
    xi = 0.1
    rho = 0.5 * xi * (Jz**2 - Jy**2)

    result = SpinSqueezing(rho, basis="Dicke", output="KU")
    expected = 0.65
    assert np.allclose(result, expected)