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

def test_bayesian_bound() -> None:
    """
    Test function for Bayesian bounds in quantum estimation.
    
    This function tests various Bayesian bounds including:
    - Bayesian Cramer-Rao Bound (BCRB)
    - Van Trees Bound (VTB)
    - Bayesian Quantum Cramer-Rao Bound (BQCRB)
    - Quantum Van Trees Bound (QVTB)
    """
    # Initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # Free Hamiltonian parameters
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    # Hamiltonian function
    hamiltonian_func = lambda x: 0.5 * b_val * omega0 * (
        sigma_x * np.cos(x) + sigma_z * np.sin(x)
    )
    
    # Derivative of Hamiltonian
    d_hamiltonian_func = lambda x: [
        0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))
    ]
    
    # Prior distribution parameters
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    mu_val, eta_val = 0.0, 0.2
    
    # Probability density function and its derivative
    prob_density = lambda x, mu, eta: np.exp(
        -(x - mu) ** 2 / (2 * eta ** 2)
    ) / (eta * np.sqrt(2 * np.pi))
    
    d_prob_density = lambda x, mu, eta: -(
        (x - mu) * np.exp(-(x - mu) ** 2 / (2 * eta ** 2))
    ) / (eta ** 3 * np.sqrt(2 * np.pi))
    
    prob_values = np.array([prob_density(x_val, mu_val, eta_val) for x_val in x_values])
    d_prob_values = np.array([d_prob_density(x_val, mu_val, eta_val) for x_val in x_values])
    
    # Normalize the distribution
    norm_factor = simpson(prob_values, x_values)
    prob_normalized = prob_values / norm_factor
    d_prob_normalized = d_prob_values / norm_factor
    
    # Time evolution parameters
    time_span = np.linspace(0.0, 1.0, 50)
    
    # Prepare arrays for states and derivatives
    final_states = []
    d_final_states = []
    
    # Evolve the system for each parameter value
    for idx in range(len(x_values)):
        hamiltonian = hamiltonian_func(x_values[idx])
        d_hamiltonian = d_hamiltonian_func(x_values[idx])
        
        dynamics = Lindblad(time_span, rho0, hamiltonian, d_hamiltonian)
        states, d_states = dynamics.expm()
        
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])  # Original structure: list of matrices

    # Test BCRB type 1
    bcrb1 = BCRB([x_values], prob_normalized, [], final_states, d_final_states, M=[], btype=1)
    expected_bcrb1 = 0.654654507602925
    assert np.allclose(bcrb1, expected_bcrb1)

    # Test BCRB type 2
    bcrb2 = BCRB([x_values], prob_normalized, [], final_states, d_final_states, M=[], btype=2)
    expected_bcrb2 = 0.651778484577857
    assert np.allclose(bcrb2, expected_bcrb2)
    
    # Test BCRB type 3
    bcrb3 = BCRB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, M=[], btype=3)
    expected_bcrb3 = 0.16522254719803486
    assert np.allclose(bcrb3, expected_bcrb3)

    # Test Van Trees Bound
    vtb = VTB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, M=[]) 
    expected_vtb = 0.03768712089828974
    assert np.allclose(vtb, expected_vtb)

    # Test BQCRB type 1
    bqcrb1 = BQCRB([x_values], prob_normalized, [], final_states, d_final_states, btype=1)
    expected_bqcrb1 = 0.5097987285760552
    assert np.allclose(bqcrb1, expected_bqcrb1)

    # Test BQCRB type 2
    bqcrb2 = BQCRB([x_values], prob_normalized, [], final_states, d_final_states, btype=2)
    expected_bqcrb2 = 0.5094351484343563
    assert np.allclose(bqcrb2, expected_bqcrb2)

    # Test BQCRB type 3
    bqcrb3 = BQCRB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, btype=3)
    expected_bqcrb3 = 0.14347116223111836
    assert np.allclose(bqcrb3, expected_bqcrb3)

    # Test Quantum Van Trees Bound
    qvtb = QVTB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states)
    expected_qvtb = 0.037087918374800306
    assert np.allclose(qvtb, expected_qvtb)
