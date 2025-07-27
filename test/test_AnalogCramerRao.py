import pytest
import numpy as np
from quanestimation.AsymptoticBound.AnalogCramerRao import (
    HCRB, 
    NHB
)

def test_HCRB_NHB():
    """
    Test Holevo Cramer-Rao bound (HCRB) and Nagaoka-Hayashi bound (NHB).
    Checks HCRB/NHB calculation for parameterized quantum state.
    """
    # Parameterized state
    theta = np.pi / 3
    phi = 0.0
    psi = np.array(
        [[np.cos(theta / 2)], 
         [np.sin(theta / 2) * np.exp(1j * phi)]]
    )
    rho = psi @ psi.conj().T
    W = np.array(
        [[1.0, 0], 
         [0, np.sin(theta) ** 2]]
    )
    
    # Density matrix derivatives
    drho_theta = np.array([
        [-np.sin(theta) / 2, np.cos(theta) * np.exp(-1j * phi) / 2],
        [np.cos(theta) * np.exp(1j * phi) / 2, np.sin(theta) / 2]
    ])
    drho_phi = np.array([
        [0, -1j * np.sin(theta) / 2 * np.exp(-1j * phi)],
        [1j * np.sin(theta) / 2 * np.exp(1j * phi), 0]
    ])
    drho = [drho_theta, drho_phi]
    
    # Calculate bounds
    result_hcrb = HCRB(rho, drho, W)
    result_nhb = NHB(rho, drho, W)
    
    # Expected results
    expected_hcrb = 4
    expected_nhb = 4
    
    # Verify calculations
    assert np.allclose(result_hcrb, expected_hcrb)
    assert np.allclose(result_nhb, expected_nhb)


def test_HCRB_NHB_invalid_input():
    """
    Test HCRB/NHB with invalid input.
    Checks TypeError raised for invalid density matrix.
    """
    # Invalid density matrix
    rho = np.array(
        [[0.5, 0.5], 
         [0.5, 0.5]]
    )
    drho_invalid = np.array(
        [[1, 2], 
         [2, 1]]
    )  # Dummy derivative
    W = np.array(
        [[1.0, 0], 
         [0, 1]]
    )  # Dummy weight matrix
    
    with pytest.raises(TypeError):
        HCRB(rho, drho_invalid, W)
    with pytest.raises(TypeError):
        NHB(rho, drho_invalid, W)

def test_HCRB_print(capfd):
    """
    Test print statements in HCRB.
    Checks correct messages printed for special cases.
    """
    rho = np.array(
        [[0.5, 0.5], 
         [0.5, 0.5]]
    )
    
    # Single parameter case
    drho1 = [np.array(
        [[1, 2], 
         [2, 1]]
    )]
    W = np.array(
        [[1.0, 0], 
         [0, 1]]
    )
    HCRB(rho, drho1, W)
    out, _ = capfd.readouterr()
    assert  "In single parameter scenario, HCRB is equivalent to QFI. Returning QFI value." in out
    
    # Rank-one weight matrix case
    drho2 = [
        np.array([[1, 0], [0, 0]]), 
        np.array([[0, 1], [1, 0]])
    ]
    W1 = np.array(
        [[1.0, 0], 
         [0, 0]]
    )  # Rank-one weight matrix
    HCRB(rho, drho2, W1)
    out, _ = capfd.readouterr()
    assert "For rank-one weight matrix, HCRB is equivalent to QFIM. Returning Tr(W @ inv(QFIM))." in out
