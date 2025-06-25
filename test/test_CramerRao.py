import pytest
from quanestimation.AsymptoticBound.CramerRao import QFIM, CFIM, QFIM_Kraus
import numpy as np

def test_CramerRao():
    """
    Test the Cramer-Rao bound for a parameterized quantum state.
    This test checks the calculation of the Quantum Fisher Information Matrix (QFIM) 
    and the Classical Fisher Information Matrix (CFIM) for a specific parameterized 
    state and its derivatives.
    """
    # parameterized state
    theta = np.pi/4
    phi = np.pi/4
    rho = np.array([[np.cos(theta)**2, np.cos(theta)*np.sin(theta)*np.exp(-1j*phi)], 
                     [np.cos(theta)*np.sin(theta)*np.exp(1j*phi), np.sin(theta)**2]])
     # derivative of the state w.r.t. theta, phi
    drho = [np.array([[-np.sin(2*theta), 2*np.cos(2*theta)*np.exp(-1j*phi)], 
                      [2*np.cos(2*theta)*np.exp(1j*phi), np.sin(2*theta)]]),
            np.array([[0, -1j*np.cos(theta)*np.sin(theta)*np.exp(-1j*phi)], 
                      [1j*np.cos(theta)*np.sin(theta)*np.exp(1j*phi), 0]])] 
    result = QFIM(rho, drho, LDtype="SLD")
    M = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]])]  # measurement operators
    resultc = CFIM(rho, drho, M)
    # check the results
    assert np.allclose(result, np.array([[4., 0.], [0., np.sin(2*theta)**2]])) == 1
    assert np.allclose(resultc, np.array([[4., 0], [0., 0.]])) == 1

def test_QFIM_Kraus():
    """
    Test the Quantum Fisher Information Matrix (QFIM) for the Kraus representation.
    This test checks the calculation of the QFIM for a specific Kraus operator.
    """
    # Kraus operator
    K0 = np.array([[1, 0], [0, np.sqrt(0.5)]])
    K1 = np.array([[np.sqrt(0.5), 0], [0, 0]])
    K = [K0, K1]  # list of Kraus operators
    # derivative of the Kraus operator w.r.t. a parameter
    dK = [[np.array([[0, 0], [0, -0.5/np.sqrt(0.5)]])], [np.array([[0, 0.5/np.sqrt(0.5)], [0, 0]])]]
    # probe state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]]) 
    # calculate the QFIM
    result = QFIM_Kraus(rho0, K, dK)
    # check the result
    assert np.allclose(result, 1.5) == 1    
