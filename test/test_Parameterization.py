# import pytest
import numpy as np
from quanestimation.Parameterization.NonDynamics import Kraus
from quanestimation.Parameterization.GeneralDynamics import Lindblad

def test_Kraus():
    """
    Test the Kraus function for quantum state evolution and derivatives.
    
    This test verifies:
    1. The evolved density matrix is computed correctly.
    2. The derivatives of the evolved density matrix with respect to 
       parameters are computed correctly.
    """
    # Kraus operators
    K0 = np.array([
        [1, 0],
        [0, np.sqrt(0.5)]
    ])
    K1 = np.array([
        [np.sqrt(0.5), 0],
        [0, 0]
    ])
    K = [K0, K1]
    
    # Kraus operator derivatives
    dK = [
        [np.array([
            [0, 0],
            [0, -0.5 / np.sqrt(0.5)]
        ])],
        [np.array([
            [0, 0.5 / np.sqrt(0.5)],
            [0, 0]
        ])]
    ]
    
    # Probe state
    rho0 = 0.5 * np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])

    rho, drho = Kraus(rho0, K, dK)
    
    expected_rho = np.array([
        [0.75, 0.35355339],
        [0.35355339, 0.25]
    ])
    expected_drho = [
        np.array([
            [0.5, -0.35355339],
            [-0.35355339, -0.5]
        ])
    ]

    assert np.allclose(rho, expected_rho)
    for i in range(len(drho)):
        assert np.allclose(drho[i], expected_drho[i])


def test_Lindblad():
    """
    Test the Lindblad function for quantum state evolution and derivatives.
    """
    # initial state
    rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    omega = 1.0
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5 * omega * sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5 * sz]
    # dissipation
    sp = np.array([[0., 1.], [0., 0.]])  
    sm = np.array([[0., 0.], [1., 0.]]) 
    decay = [[sp, 0.0], [sm, 0.1]]
    # time length for the evolution
    tspan = np.linspace(0., 1., 20)
    # dynamics
    dynamics = Lindblad(tspan, rho0, H0, dH, decay)
    rho, drho = dynamics.expm()
    expected_rho = np.array([
        [0.45241871+0.j, 0.25697573-0.40021598j],
        [0.25697573+0.40021598j, 0.54758129+0.j]])
    drho_final = drho[-1]
    expected_drho = [np.array([
        [0.+0.j, -0.40021598-0.25697573j],
        [-0.40021598+0.25697573j, 0.+0.j]])]
    assert np.allclose(rho[-1], expected_rho)
    for i in range(len(drho_final)):
        assert np.allclose(drho_final[i], expected_drho[i])