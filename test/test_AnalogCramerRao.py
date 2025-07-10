import pytest
import numpy as np
from quanestimation.AsymptoticBound.AnalogCramerRao import HCRB, NHB 
from quanestimation.Parameterization.GeneralDynamics import Lindblad

def test_HCRB_NHB():
    """
    Test the Holevo Cramer-Rao bound (HCRB) and Nagaoka-Hayashi bound (NHB) for a parameterized quantum state.
    This test checks the calculation of the HCRB and NHB for a specific density matrix and its derivatives.
    """
    # initial state
    psi0 = np.array([1., 0., 0., 1.])/np.sqrt(2)
    rho0 = psi0.reshape(-1,1) @ psi0.reshape(1,-1).conj()
    # free Hamiltonian
    omega1, omega2, g = 1.0, 1.0, 0.1
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]]) 
    sz = np.array([[1., 0.], [0., -1.]])
    ide = np.array([[1., 0.], [0., 1.]])   
    H0 = omega1*np.kron(sz, ide)+omega2*np.kron(ide, sz)+g*np.kron(sx, sx)
    # derivatives of the free Hamiltonian on omega2 and g
    dH = [np.kron(ide, sz), np.kron(sx, sx)] 
    # time length for the evolution
    tnum = 10
    tspan = np.linspace(0., 0.1, tnum)
    # dynamics
    dynamics = Lindblad(tspan, rho0, H0, dH)
    rho, drho = dynamics.expm()
    # weight matrix
    W = ide
    # calculation of the HCRB and NHB
    f_HCRB, f_NHB = [], []
    for ti in range(1, tnum):
        # HCRB
        f_tp1 = HCRB(rho[ti], drho[ti], W, eps=1e-6)
        f_HCRB.append(f_tp1)
        # NHB
        f_tp2 = NHB(rho[ti], drho[ti], W)
        f_NHB.append(f_tp2)
    
    expected_HCRB = 83.97255661077813
    expected_NHB = 565.0877925867512
    # Check if the result is greater than or equal to zero
    assert np.allclose(f_HCRB[-1], expected_HCRB) == 1
    assert np.allclose(f_NHB[-1], expected_NHB) == 1