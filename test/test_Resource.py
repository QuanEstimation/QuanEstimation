import pytest
from quanestimation.Resource.Resource import (
    SpinSqueezing, 
    TargetTime
)
import numpy as np

def test_SpinSqueezing_Dicke():
    """
    Test the SpinSqueezing function with valid input for Dicke basis 
    and check exception handling.
    
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

    result1 = SpinSqueezing(density_matrix, basis="Dicke", output="WBIMH")
    expected1 = 8.0753
    assert np.allclose(result1, expected1)

    # Test invalid output type
    with pytest.raises(ValueError):
        SpinSqueezing(density_matrix, basis="Dicke", output="invalid")

    with pytest.raises(ValueError):
        SpinSqueezing(density_matrix, basis="invalid", output="KU")    

def test_SpinSqueezing_Pauli():
    """
    Test the SpinSqueezing function with Pauli basis.
    
    This test verifies:
        The function returns the expected value for valid input.
    """
    # Collective spin operators for j=2 system in Pauli basis
    # sy = np.array([[0., -1j], [1j, 0.]])
    # sz = np.array([[1., 0.], [0., -1.]])
    # ide = np.identity(2)

    # jy_matrix = np.kron(ide, np.kron(sy, ide)) + np.kron(sy, np.kron(ide, ide)) + np.kron(ide, np.kron(ide, sy)) 
    # jz_matrix = np.kron(ide, np.kron(sz, ide)) + np.kron(sz, np.kron(ide, ide)) + np.kron(ide, np.kron(ide, sz))
    
    # xi_param = 0.1
    # density_matrix = 0.5 * xi_param * (jz_matrix**2 - jy_matrix**2)
    a = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    density_matrix = np.diag(a)

    # Test valid output type
    result = SpinSqueezing(density_matrix, basis="Pauli", output="KU")
    expected = 1.
    assert np.allclose(result, expected)

def test_SpinSqueezing_nomean():
    """
    Test the SpinSqueezing function with a density matrix that has no mean values of Jx, Jy, and Jz.
    
    This test verifies:
        The function raises ValueError when the density matrix does not have a valid spin squeezing.
    """
    # Create a density matrix with no mean values
    rho = np.array([[0.5, 0.0], [0.0, 0.5]])
    
    with pytest.raises(ValueError):
        SpinSqueezing(rho, basis="Pauli", output="KU")


def test_TargetTime():
    """
    Test the TargetTime function.

    This test verifies:
        The function returns the expected value for valid input.
    """

    testfunc = lambda t, omega: np.cos(omega * t)
    tspan = np.linspace(0, np.pi, 1000000)
    target = 0.  
    result = TargetTime(target, tspan, testfunc, 1.)
    expected = np.pi / 2
    assert np.allclose(result, expected, atol=1e-4)

def test_TargetTime_no_crossing(capfd):
    """
    Test the TargetTime function when no crossing occurs.

    This test verifies:
        The function returns None when no crossing is found.
    """
    
    testfunc = lambda t: np.cos(t)
    tspan = np.linspace(0, 1, 100)
    target = 2.  # No crossing with cos(t)
    
    result = TargetTime(target, tspan, testfunc)

    out, _ = capfd.readouterr()
    assert  "No time is found in the given time span to reach the target." in out
    assert result is None    
