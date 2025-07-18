import pytest
from quanestimation.Resource.Resource import SpinSqueezing
import numpy as np

def test_SpinSqueezing():
    """
    Test the SpinSqueezing function with valid input and check exception handling.
    
    This test verifies:
    1. The function returns the expected value for valid input
    2. The function raises NameError for invalid output type
    """
    # Collective spin operators for j=2 system
    jy_matrix = np.array(
        [
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 - 1.22474487j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.22474487j, 0.0 + 0.0j, 0.0 - 1.22474487j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.22474487j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j]
        ]
    )
    
    jz_matrix = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0]
        ]
    )
    
    xi_param = 0.1
    density_matrix = 0.5 * xi_param * (jz_matrix**2 - jy_matrix**2)

    # Test valid output type
    result = SpinSqueezing(density_matrix, basis="Dicke", output="KU")
    expected = 0.65
    assert np.allclose(result, expected)

    # Test invalid output type
    with pytest.raises(NameError):
        SpinSqueezing(density_matrix, basis="Dicke", output="invalid")

    with pytest.raises(ValueError):
        SpinSqueezing(density_matrix, basis="invalid", output="KU")    
