import pytest
import numpy as np
from quanestimation.AsymptoticBound.AnalogCramerRao import HCRB, NHB 

def test_HCRB_NHB():
    """
    Test the Holevo Cramer-Rao bound (HCRB) and Nagaoka-Hayashi bound (NHB) for a parameterized quantum state.
    This test checks the calculation of the HCRB and NHB for a specific density matrix and its derivatives.
    """
    # parameterized state
    theta = np.pi/3
    phi = 0.
    psi = np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j * phi)]])
    rho = psi @ psi.conj().T
    W = np.array([[1., 0], [0, np.sin(theta)**2]])
    # derivatives of the density matrix w.r.t. theta and phi
    drho_theta = np.array([[-np.sin(theta)/2, np.cos(theta)*np.exp(-1j*phi)/2],
                           [np.cos(theta)*np.exp(1j*phi)/2, np.sin(theta)/2]])
    drho_phi = np.array([[0, -1j*np.sin(theta)/2 * np.exp(-1j*phi)],
                         [1j*np.sin(theta)/2 * np.exp(1j*phi), 0]])
    drho = [drho_theta, drho_phi]
    # calculate the Holevo Cramer-Rao bound (HCRB) and Nagaoka-Hayashi bound (NHB)
    result_HCRB = HCRB(rho, drho, W)
    result_NHB = NHB(rho, drho, W)
    # expected results
    expected_HCRB = 4
    expected_NHB = 4
    # check if the results match the expected values
    assert np.allclose(result_HCRB, expected_HCRB) == 1
    assert np.allclose(result_NHB, expected_NHB) == 1

def test_HCRB_NHB_invalid_input():
    """
    Test the Holevo Cramer-Rao bound (HCRB) and Nagaoka-Hayashi bound (NHB) with invalid input.
    This test checks if the functions raise a ValueError when provided with an invalid density matrix.
    """
    # invalid density matrix (not positive semi-definite)
    rho = np.array([[0.5, 0.5], [0.5, 0.5]])
    drho_invalid = np.array([[1, 2], [2, 1]])  # dummy derivative
    W = np.array([[1., 0], [0, 1]])  # dummy weight matrix
    with pytest.raises(TypeError):
        HCRB(rho, drho_invalid, W)
    with pytest.raises(TypeError):
        NHB(rho, drho_invalid, W)

def test_HCRB_print(capfd):
    """
    Test the print statements in the Holevo Cramer-Rao bound (HCRB)
    This test checks if the function prints the correct message when a single parameter is provided.
    """
    rho = np.array([[0.5, 0.5], [0.5, 0.5]])
    drho1 = [np.array([[1, 2], [2, 1]])] 
    W = np.array([[1., 0], [0, 1]])
    HCRB(rho, drho1, W)
    out, _ = capfd.readouterr()
    assert "In single parameter scenario, HCRB is equivalent to QFI. This function will return the value of QFI." in out

    drho2 = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])]  # two parameters
    W1 = np.array([[1., 0], [0, 0]])  # rank-one weight matrix
    HCRB(rho, drho2, W1)
    out, _ = capfd.readouterr()
    assert "For rank-one weight matrix, the HCRB is equivalent to QFIM. This function will return the value of Tr(WF^{-1})." in out

    