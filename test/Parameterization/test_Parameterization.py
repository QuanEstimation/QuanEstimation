# import pytest
import pytest
import numpy as np
from quanestimation.base.Parameterization.NonDynamics import Kraus
from quanestimation.base.Parameterization.GeneralDynamics import Lindblad

def test_Kraus() -> None:
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


def test_Lindblad() -> None:
    """
    Test the Lindblad function for quantum state evolution and derivatives.
    
    This test verifies:
    - Correct evolution of a quantum state under Lindblad dynamics
    - Proper calculation of parameter derivatives
    - Handling of dissipation effects with decay operators
    
    Test scenario: Two-level system with spontaneous emission.
    """
    # Initial state
    initial_state = 0.5 * np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    
    # Free Hamiltonian parameters
    frequency = 1.0
    sz = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ])
    hamiltonian = 0.5 * frequency * sz
    
    # Derivative of Hamiltonian with respect to frequency
    hamiltonian_derivative = [0.5 * sz]
    
    # Dissipation operators
    sp = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ])  
    sm = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ]) 
    decay_operators = [[sp, 0.0], [sm, 0.1]]
    
    # Time points for evolution
    tspan = np.linspace(0.0, 1.0, 10)

    # control Hamiltonians and coefficients
    sx = np.array([[0., 1.], [1., 0.]])
    control_amplitudes = np.zeros(len(tspan))
    
    # Create Lindblad dynamics
    dynamics = Lindblad(
        tspan, 
        initial_state, 
        hamiltonian, 
        hamiltonian_derivative, 
        decay_operators, 
        Hc=[sx],
        ctrl=[control_amplitudes]
    )
    final_state_expm, state_derivatives_expm = dynamics.expm()
    final_state_ode, state_derivatives_ode = dynamics.ode()
    
    # Expected final state
    expected_final_state = np.array([
        [0.45241871 + 0.j, 0.25697573 - 0.40021598j],
        [0.25697573 + 0.40021598j, 0.54758129 + 0.j]
    ])
    assert np.allclose(final_state_expm[-1], expected_final_state, atol = 1e-6)
    assert np.allclose(final_state_ode[-1], expected_final_state, atol = 1e-6)

    # Expected derivative of final state
    final_state_derivative_expm = state_derivatives_expm[-1]
    final_state_derivative_ode = state_derivatives_ode[-1]
    expected_derivative_expm = [
        np.array([
            [0.0 + 0.j, -0.40021598 - 0.25697573j],
            [-0.40021598 + 0.25697573j, 0.0 + 0.j]
        ])
    ]
    expected_derivative_ode = [
        np.array([
            [0.+0.j, -0.40182466-0.25365255j],
            [-0.40182466+0.25365255j, 0.+0.j]])
    ]


    for i in range(len(final_state_derivative_expm)):
        assert np.allclose(
            final_state_derivative_expm[i], 
            expected_derivative_expm[i], 
            atol=1e-6
        )

    for i in range(len(final_state_derivative_ode)):
        assert np.allclose(
            final_state_derivative_ode[i], 
            expected_derivative_ode[i], 
            atol=1e-6
        )

    with pytest.raises(TypeError):
            dynamics = Lindblad(
        tspan, 
        initial_state, 
        hamiltonian, 
        np.array([0., 1.]),  # Incorrect type for derivative
        decay_operators
    )
            
# def test_Lindblad_secondorder_derivative():
#     """
#     Test the second-order derivative of the Lindblad dynamics.
    
#     This test verifies:
#         - the second-order derivative of the state under the Lindblad dynamics.   

#     Test scenario: Two-level system with spontaneous emission.
#     """
#     # Initial state
#     initial_state = 0.5 * np.array([
#         [1.0, 1.0],
#         [1.0, 1.0]
#     ])
    
#     # Free Hamiltonian parameters
#     frequency = 1.0
#     sz = np.array([
#         [1.0, 0.0],
#         [0.0, -1.0]
#     ])
#     hamiltonian = 0.5 * frequency * sz

#     # Derivative of Hamiltonian with respect to frequency
#     hamiltonian_derivative = [0.5 * sz]
#     hamiltonian_second_derivative = [np.zeros((2, 2))]
    
#     # Dissipation operators
#     sp = np.array([
#         [0.0, 1.0],
#         [0.0, 0.0]
#     ])  
#     sm = np.array([
#         [0.0, 0.0],
#         [1.0, 0.0]
#     ]) 
#     decay_operators = [[sp, 0.0], [sm, 0.1]]
    
#     # Time points for evolution
#     tspan = np.linspace(0.0, 1.0, 10)

#     # control Hamiltonians and coefficients
#     sx = np.array([[0., 1.], [1., 0.]])
#     control_amplitudes = np.zeros(len(tspan))
    
#     # Create Lindblad dynamics
#     dynamics = Lindblad(
#         tspan, 
#         initial_state, 
#         hamiltonian, 
#         hamiltonian_derivative, 
#         decay_operators, 
#         Hc=[sx],
#         ctrl=[control_amplitudes]
#     )
#     final_state, state_derivatives, state_second_derivative = dynamics.secondorder_derivative(hamiltonian_second_derivative)
    
#     # Expected final state
#     expected_final_state = np.array([
#         [0.45241871 + 0.j, 0.25697573 - 0.40021598j],
#         [0.25697573 + 0.40021598j, 0.54758129 + 0.j]
#     ])
#     assert np.allclose(final_state[-1], expected_final_state, atol=1e-6)


# --- Phase 4.11 regression tests (Julia @testset parity) ---


def test_41_ode_tolerance() -> None:
    """#41: Expm and Ode dynamics must agree within 1e-2.

    Source: test_lindblad.jl @testset "#41"
    """
    rho0 = np.eye(2, dtype=np.complex128) * 0.5
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H0 = 0.5 * sz
    dH = [0.5 * sz]
    tspan = np.linspace(0.0, 0.5, 20)
    decay = [[sx, 0.1]]

    dyn_e = Lindblad(tspan, rho0, H0, dH, decay)
    rho_e, _ = dyn_e.expm()

    dyn_o = Lindblad(tspan, rho0, H0, dH, decay)
    rho_o, _ = dyn_o.ode()

    diff = np.linalg.norm(rho_e[-1] - rho_o[-1])
    assert diff < 1e-2


def test_42_zero_decay_threshold() -> None:
    """#42: Near-zero decay rate must not break density matrix properties.

    Source: test_lindblad.jl @testset "#42"
    """
    rho0 = np.eye(2, dtype=np.complex128) * 0.5
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H0 = 0.5 * sz
    dH = [0.5 * sz]
    tspan = np.linspace(0.0, 0.5, 20)
    decay = [[sx, 0.0], [sz, 1e-8]]

    dyn = Lindblad(tspan, rho0, H0, dH, decay)
    rho, _ = dyn.expm()
    for r in rho:
        assert np.all(np.linalg.eigvalsh(r) >= -1e-14)
        assert np.allclose(r, r.conj().T, atol=1e-14)
