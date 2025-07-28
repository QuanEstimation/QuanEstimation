# import pytest
import numpy as np
from scipy.integrate import simpson
from quanestimation.BayesianBound.ZivZakai import QZZB
from quanestimation.Parameterization.GeneralDynamics import Lindblad

def test_ZivZakai():
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
    p_tp = [p_func(x[i], mu, eta) for i in range(len(x))]
    # normalization of the distribution
    c = simpson(p_tp, x)
    p = p_tp/c 
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

    f_QZZB = QZZB([x], p, rho)
    expected_QZZB = 0.028521709437588784 
    assert np.allclose(f_QZZB, expected_QZZB)  

    
