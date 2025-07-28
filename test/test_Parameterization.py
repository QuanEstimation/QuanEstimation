# import pytest
import numpy as np
from quanestimation.Parameterization.NonDynamics import Kraus

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
