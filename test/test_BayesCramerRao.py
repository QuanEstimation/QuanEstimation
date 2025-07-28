# import pytest
import numpy as np
from scipy.integrate import simpson
from quanestimation.BayesianBound.BayesCramerRao import (
    BCRB, 
    VTB, 
    BQCRB, 
    QVTB
)
from quanestimation.Parameterization.GeneralDynamics import Lindblad

def test_BayesianBound():
    # initial state
    rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    B, omega0 = 0.5 * np.pi, 1.0
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    H0_func = lambda x: 0.5 * B * omega0 * (sx * np.cos(x) + sz * np.sin(x))
    # derivative of the free Hamiltonian on x
    dH_func = lambda x: [0.5 * B * omega0 * (-sx * np.sin(x) + sz * np.cos(x))]
    # prior distribution
    x = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    mu, eta = 0.0, 0.2
    p_func = lambda x, mu, eta: np.exp(-(x-mu)**2 / (2 * eta**2))/(eta * np.sqrt(2 * np.pi))
    dp_func = lambda x, mu, eta: -(x-mu) * np.exp(-(x-mu)**2/(2 * eta**2))/(eta**3 * np.sqrt(2*np.pi))
    p_tp = [p_func(x[i], mu, eta) for i in range(len(x))]
    dp_tp = [dp_func(x[i], mu, eta) for i in range(len(x))]
    # normalization of the distribution
    c = simpson(p_tp, x)
    p, dp = p_tp/c, dp_tp/c
    # time length for the evolution
    tspan = np.linspace(0., 1., 50)
    # dynamics
    rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x))]
    drho = [[np.zeros((len(rho0), len(rho0)), dtype=np.complex128)] for i in range(len(x))]
    for i in range(len(x)):
        H0_tp = H0_func(x[i])
        dH_tp = dH_func(x[i])
        dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[i] = rho_tp[-1]
        drho[i] = drho_tp[-1] 

    f_BCRB1 = BCRB([x], p, [], rho, drho, M=[], btype=1)
    expected_BCRB1 = 0.654654507602925
    assert np.allclose(f_BCRB1, expected_BCRB1)

    f_BCRB2 = BCRB([x], p, [], rho, drho, M=[], btype=2)
    expected_BCRB2 = 0.651778484577857
    assert np.allclose(f_BCRB2, expected_BCRB2)
    
    f_BCRB3 = BCRB([x], p, dp, rho, drho, M=[], btype=3)
    expected_BCRB3 = 0.16522254719803486
    assert np.allclose(f_BCRB3, expected_BCRB3)

    f_VTB = VTB([x], p, dp, rho, drho, M=[]) 
    expected_VTB = 0.03768712089828974   
    assert np.allclose(f_VTB, expected_VTB)

    f_BQCRB1 = BQCRB([x], p, [], rho, drho, btype=1)
    expected_BQCRB1 = 0.5097987285760552
    assert np.allclose(f_BQCRB1, expected_BQCRB1)

    f_BQCRB2 = BQCRB([x], p, [], rho, drho, btype=2)
    expected_BQCRB2 = 0.5094351484343563
    assert np.allclose(f_BQCRB2, expected_BQCRB2)

    f_BQCRB3 = BQCRB([x], p, dp, rho, drho, btype=3)
    expected_BQCRB3 = 0.14347116223111836
    assert np.allclose(f_BQCRB3, expected_BQCRB3)

    f_QVTB = QVTB([x], p, dp, rho, drho)
    expected_QVTB = 0.037087918374800306
    assert np.allclose(f_QVTB, expected_QVTB)