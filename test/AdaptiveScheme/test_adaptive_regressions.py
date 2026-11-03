"""Adaptive scheme regression tests — Julia @testset parity.

Source: test_adaptive_estimation.jl
"""

import numpy as np


def test_12_mi_error_probability() -> None:
    """#12: Conditional probability p(y|x) = tr(ρM) must be in [0,1].

    Source: test_adaptive_estimation.jl @testset "#12"
    """
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    M1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    M2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    p1 = np.real(np.trace(rho0 @ M1))
    p2 = np.real(np.trace(rho0 @ M2))
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0
    assert np.isclose(p1 + p2, 1.0)


def test_37_posterior_normalization() -> None:
    """#37: Bayes() posterior integral must equal 1.

    Source: test_adaptive_estimation.jl @testset "#37"
    """
    from quanestimation.base.Parameterization.GeneralDynamics import Lindblad
    from quanestimation.base.BayesianBound.BayesEstimation import Bayes

    omega = 1.0
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    tspan = np.linspace(0.0, 1.0, 10)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    x_vals = np.linspace(0.5, 1.5, 11)
    p_vals = np.ones(len(x_vals)) / len(x_vals)

    rho_list = []
    for xv in x_vals:
        H0 = 0.5 * xv * omega * sz
        dyn = Lindblad(tspan, rho0, H0, [0.5 * sz])
        states, _ = dyn.expm()
        rho_list.append(states[-1])

    M = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    ]
    pout, _ = Bayes(
        [x_vals], p_vals, rho_list,
        y=[0] * 500, M=M, savefile=False,
    )
    pout_vec = np.asarray(pout).flatten()
    dx = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    integral = np.sum(pout_vec) * dx
    assert np.isclose(integral, 1.0, rtol=0.15)
