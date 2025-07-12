import pytest
import numpy as np
from quanestimation.AsymptoticBound.AnalogCramerRao import HCRB, NHB 
from quanestimation.Parameterization.GeneralDynamics import Lindblad

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
    sz = np.array([[1, 0], [0, -1]])
    sy = np.array([[0, -1j], [1j, 0]])   
    W = np.array([[1., 0], [0, np.sin(theta)**2]])
    # derivatives of the density matrix w.r.t. theta and phi
    drho_theta = np.array([[-np.sin(theta)/2, np.cos(theta)*np.exp(-1j*phi)/2],
                           [np.cos(theta)*np.exp(1j*phi)/2, np.sin(theta)/2]])
    drho_phi = np.array([[0, -1j*np.sin(theta)/2 * np.exp(-1j*phi)],
                         [1J*np.sin(theta)/2 * np.exp(1j*phi), 0]])
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