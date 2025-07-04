import pytest
from quanestimation.AsymptoticBound.CramerRao import QFIM, CFIM, QFIM_Kraus, QFIM_Bloch, QFIM_Gauss, LLD, RLD
import numpy as np

def test_CramerRao_SLD():
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
    # list of Kraus operators
    K = [K0, K1]  
    # derivatives of the Kraus operator w.r.t. the parameter
    dK = [[np.array([[0, 0], [0, -0.5/np.sqrt(0.5)]])], [np.array([[0, 0.5/np.sqrt(0.5)], [0, 0]])]]
    # probe state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]]) 
    # calculate the QFIM
    result = QFIM_Kraus(rho0, K, dK)
    # check the result
    assert np.allclose(result, 1.5) == 1    

def test_QFIM_Bloch():
    """
    Test the Quantum Fisher Information Matrix (QFIM) for the Bloch vector representation.
    This test checks the calculation of the QFIM for a specific Bloch vector and its derivatives.
    """
    # Bloch vector
    theta = np.pi/4
    phi = np.pi/2
    eta = 0.8
    b = eta*np.array([np.sin(2*theta)*np.cos(phi), np.sin(2*theta)*np.sin(phi), np.cos(2*theta)])
    # derivatives of the Bloch vector w.r.t. theta, phi
    db_theta = eta*np.array([2*np.cos(2*theta)*np.cos(phi), 2*np.cos(2*theta)*np.sin(phi), -2*np.sin(2*theta)])
    db_phi = eta*np.array([-np.sin(2*theta)*np.sin(phi), np.sin(2*theta)*np.cos(phi), 0])
    # list of derivatives
    db = [db_theta, db_phi] 
    # calculate the QFIM
    result = QFIM_Bloch(b, db)
    # check the result
    assert np.allclose(result, np.array([[4.*eta**2, 0.], [0., eta**2*np.sin(2*theta)**2]])) == 1

def test_QFIM_Gauss():
    """
    Test the Quantum Fisher Information Matrix (QFIM) for the Gaussian state representation.
    This test checks the calculation of the QFIM for a specific Gaussian state and its derivatives.
    The example is the Eq. (38) in J. Phys. A: Math. Theor. 52, 035304 (2019). The analytical result 
    is wrong in the paper, and the right one is F = [[(lamb*lamb-1)**2/2/(4*lamb**2-1), 0.], [0., 8*lamb*lamb/(4*lamb*lamb+1)]]. 
    """
    # Gaussian state parameters
    r = 0.8  # squeezing parameter
    beta = 0.5
    lamb = 1/np.tanh(beta/2)  # lambda parameter
    mu = np.array([0., 0.])  # mean vector
    sigma = lamb*np.array([[np.cosh(2*r), -np.sinh(2*r)], [-np.sinh(2*r), np.cosh(2*r)]])  # covariance matrix
    # derivatives of the Gaussian state w.r.t. the parameters
    dmu = [np.array([0., 0.]), np.array([0., 0.])]  # derivatives of mean vector
    dlamb = -0.5/(np.sinh(beta/2)**2)  # derivative of lambda w.r.t. beta
    dsigma = [dlamb*np.array([[np.cosh(2*r), -np.sinh(2*r)], [-np.sinh(2*r), np.cosh(2*r)]]), 
              lamb*2*np.array([[np.sinh(2*r), -np.cosh(2*r)], [-np.cosh(2*r), np.sinh(2*r)]])] # derivatives of covariance matrix
    # calculate the QFIM
    result = QFIM_Gauss(mu, dmu, sigma, dsigma)
    # check the result
    assert np.allclose(result, np.array([[(lamb*lamb-1)**2/2/(4*lamb**2-1), 0.], [0., 8*lamb*lamb/(4*lamb*lamb+1)]])) == 1  

def test_QFIM_LLD():
    """
    Test the left logarithmic derivative (LLD) for a specific parameterized quantum state.
    This test checks the calculation of the LLD for a specific state and its derivatives.
    """
    # parameterized state
    theta = np.pi/4
    phi = np.pi/4
    eta = 0.8
    rho = 0.5*np.array([[1+eta*np.cos(2*theta), eta*np.sin(2*theta)*np.exp(-1j*phi)], 
                     [eta*np.sin(2*theta)*np.exp(1j*phi), 1-eta*np.cos(2*theta)]])
    # derivative of the state w.r.t. phi
    drho = [0.5*np.array([[0., -1j*eta*np.sin(2*theta)*np.exp(-1j*phi)],[1j*eta*np.sin(2*theta)*np.exp(1j*phi), 0.]])] 
    # calculate the LLD
    result = LLD(rho, drho, rep="original")
    expected = (1/(1-eta**2))*np.array([[1j*eta**2*np.sin(2*theta)**2, -1j*eta*(1+eta*np.cos(2*theta))*np.sin(2*theta)*np.exp(-1j*phi)], 
                                        [1j*eta*(1-eta*np.cos(2*theta))*np.sin(2*theta)*np.exp(1j*phi), -1j*eta**2*np.sin(2*theta)**2]])
    
    result_QFIM = QFIM(rho, drho, LDtype="LLD")
    expected_QFIM = eta**2*np.sin(2*theta)**2*(3*eta**2+1)/(1-eta**2)**2

    # check the result
    assert np.allclose(result, expected) == 1
    assert np.allclose(result_QFIM, expected_QFIM) == 1

def test_QFIM_RLD():
    """
    Test the right logarithmic derivative (RLD) for a specific parameterized quantum state.
    This test checks the calculation of the RLD for a specific state and its derivatives.
    """
    # parameterized state
    theta = np.pi/4
    phi = np.pi/4
    eta = 0.8
    rho = 0.5*np.array([[1+eta*np.cos(2*theta), eta*np.sin(2*theta)*np.exp(-1j*phi)], 
                     [eta*np.sin(2*theta)*np.exp(1j*phi), 1-eta*np.cos(2*theta)]])
    # derivative of the state w.r.t. phi
    drho = [0.5*np.array([[0., -1j*eta*np.sin(2*theta)*np.exp(-1j*phi)],[1j*eta*np.sin(2*theta)*np.exp(1j*phi), 0.]])]
    # calculate the RLD
    result = RLD(rho, drho, rep="original")
    expected = (1/(1-eta**2))*np.array([[-1j*eta**2*np.sin(2*theta)**2, -1j*eta*(1-eta*np.cos(2*theta))*np.exp(-1j*phi)*np.sin(2*theta)], 
                                        [1j*eta*(1+np.cos(2*theta))*np.exp(1j*phi)*np.sin(2*theta), 1j*eta**2*np.sin(2*theta)**2]])
    # check the result
    result_QFIM = QFIM(rho, drho, LDtype="RLD")
    expected_QFIM = eta**2*np.sin(2*theta)**2/(1.-eta**2)

    assert np.allclose(result, expected) == 1
    assert np.allclose(result_QFIM, expected_QFIM) == 1
