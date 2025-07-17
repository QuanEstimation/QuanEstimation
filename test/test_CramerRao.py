import pytest
from quanestimation.AsymptoticBound.CramerRao import QFIM, CFIM, QFIM_Kraus, QFIM_Bloch, QFIM_Gauss, LLD, RLD, FIM, FI_Expt, SLD
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

    result, SLD_res = QFIM(rho, drho, LDtype="SLD", exportLD=True)
    expected_SLD = SLD(rho, drho)
    assert np.allclose(SLD_res, expected_SLD) == 1

    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="invalid")

    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="RLD")  
    
    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="LLD")    

def test_CFIM_singleparameter():
    """
    Test the Classical Fisher Information Matrix (CFIM) for a single parameter.
    This test checks the calculation of the CFIM for a specific state and its derivatives.
    """
    # parameterized state
    theta = np.pi/4
    rho = np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)], 
                     [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
    # derivative of the state w.r.t. theta
    drho = [np.array([[-np.sin(2*theta), 2*np.cos(2*theta)], 
                      [2*np.cos(2*theta), np.sin(2*theta)]])]
    # calculate the CFIM
    result = CFIM(rho, drho, [])
    # check the result
    assert np.allclose(result, 2) == 1

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

def test_QFIM_Gauss_multiparameter():
    """
    Test the Quantum Fisher Information Matrix (QFIM) for the Gaussian state representation in the case 
    of multiparameter estimation. This test checks the calculation of the QFIM for a specific Gaussian 
    state and its derivatives. The example is the Eq. (38) in J. Phys. A: Math. Theor. 52, 035304 (2019). 
    The analytical result is wrong in the paper, and the right one is F = [[(lamb*lamb-1)**2/2/(4*lamb**2-1), 0.], 
    [0., 8*lamb*lamb/(4*lamb*lamb+1)]]. 
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

def test_QFIM_Gauss_singleparameter():
    """
    Test the Quantum Fisher Information Matrix (QFIM) for the Gaussian state representation in the case 
    of single parameter estimation. This test checks the calculation of the QFIM for a specific Gaussian 
    state and its derivatives. The example is the Eq. (38) in J. Phys. A: Math. Theor. 52, 035304 (2019). 
    The analytical result is wrong in the paper, and the right one is F = [[(lamb*lamb-1)**2/2/(4*lamb**2-1), 0.], 
    [0., 8*lamb*lamb/(4*lamb*lamb+1)]]. 
    """
    # Gaussian state parameters
    r = 0.8  # squeezing parameter
    beta = 0.5
    lamb = 1/np.tanh(beta/2)  # lambda parameter
    mu = np.array([0., 0.])  # mean vector
    sigma = lamb*np.array([[np.cosh(2*r), -np.sinh(2*r)], [-np.sinh(2*r), np.cosh(2*r)]])  # covariance matrix
    # derivatives of the Gaussian state w.r.t. the parameters
    dmu = [np.array([0., 0.])]  # derivatives of mean vector
    dlamb = -0.5/(np.sinh(beta/2)**2)  # derivative of lambda w.r.t. beta
    dsigma = [dlamb*np.array([[np.cosh(2*r), -np.sinh(2*r)], [-np.sinh(2*r), np.cosh(2*r)]])] # derivatives of covariance matrix
    # calculate the QFIM
    result = QFIM_Gauss(mu, dmu, sigma, dsigma)
    # check the result
    assert np.allclose(result, (lamb*lamb-1)**2/2/(4*lamb**2-1)) == 1     

def test_QFIM_LLD_singleparameter():
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

    result_eigen = LLD(rho, drho, rep="eigen")
    _, vec = np.linalg.eig(rho)
    expected_eigen =  vec.conj().transpose() @ result @ vec
    assert np.allclose(result_eigen, expected_eigen) == 1

    with pytest.raises(ValueError):
        LLD(rho, drho, rep="invalid")

def test_QFIM_RLD_singleparameter():
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

    result_eigen = RLD(rho, drho, rep="eigen")
    _, vec = np.linalg.eig(rho)
    expected_eigen =  vec.conj().transpose() @ result @ vec
    assert np.allclose(result_eigen, expected_eigen) == 1

    with pytest.raises(ValueError):
        RLD(rho, drho, rep="invalid")

def test_FIM():
    """
    Test the calculation of the Fisher Information Matrix (FIM) for classical scenarios.
    This test checks the calculation of the FIM for classical scenarios.
    """
    x = 1.
    theta = np.pi/3
    p = np.array([np.cos(x*theta)**2, np.sin(x*theta)**2])
    dp = [np.array([-x*np.sin(2*x*theta), x*np.sin(2*x*theta)])]
    # calculate the FIM
    result = FIM(p,dp)
    # check the result
    assert np.allclose(result, 4) == 1    

def test_FIM_multiparameter():
    """
    Test the Fisher Information Matrix (FIM) for a single parameter.
    This test checks the calculation of the FIM for a specific state and its derivatives.
    """
    x = 1.
    theta = np.pi/3
    p = np.array([np.cos(x*theta)**2, np.sin(x*theta)**2])
    dp = [np.array([-theta*np.sin(2*x*theta), theta*np.sin(2*x*theta)]), np.array([-x*np.sin(2*x*theta), x*np.sin(2*x*theta)])]
    # calculate the FIM
    result = FIM(p,dp)
    expected = np.array([[4.38649084, 4.1887902], [4.1887902, 4.]])
    # check the result
    assert np.allclose(result, expected) == 1      

def test_FI_Expt():
    """
    Test the calculation of the Fisher Information for a specific experiment data.
    This test checks the calculation of the Fisher Information for a specific experiment.
    """
    dx = 0.001
    y1 = np.random.normal(loc=0.0, scale=1.0, size=1000)
    y2 = np.random.normal(loc=dx, scale=1.0, size=1000)
    # calculate the Fisher Information 
    result1 = FI_Expt(y1, y2, dx, ftype="norm")
    result2 = FI_Expt(y1, y2, dx, ftype="gamma")
    result3 = FI_Expt(y1, y2, dx, ftype="rayleigh")
    result4 = FI_Expt(y1, y2, dx, ftype="poisson")
    # check the result is a float and approximately 1.0
    assert isinstance(result1, float)
    assert isinstance(result2, float)
    assert isinstance(result3, float)
    assert isinstance(result4, float)

    with pytest.raises(ValueError):
        FI_Expt(y1, y2, dx, ftype="invalid")

def test_invalid_input():
    """
    Test the input validation for the functions in the Cramer-Rao module.
    This test checks that the functions raise appropriate errors for invalid inputs.
    """
    with pytest.raises(TypeError):
        CFIM(np.array([[1, 0], [0, 1]]), None, None) # Invalid input type   

    with pytest.raises(TypeError):
        CFIM(np.array([[1, 0], [0, 1]]), [np.array([[1, 0], [0, 1]])], None) # Invalid input type     

    with pytest.raises(ValueError):
        QFIM(np.array([[1, 0], [0, 1]]), [np.array([[1, 0], [0, 1]])], LDtype="invalid") # Invalid input type

    with pytest.raises(TypeError):
        QFIM(np.array([[1, 0], [0, 1]]), None) # Invalid input type         

    with pytest.raises(TypeError):
        QFIM_Bloch(np.array([[1.], [0], [1.]]), None) # Invalid input type   

    with pytest.raises(TypeError):  
        SLD(np.array([[1, 0], [0, 1]]), None) # Invalid input type      

    with pytest.raises(TypeError):  
        LLD(np.array([[1, 0], [0, 1]]), None) # Invalid input type   

    with pytest.raises(TypeError):  
        RLD(np.array([[1, 0], [0, 1]]), None) # Invalid input type     


