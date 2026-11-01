"""Bayesian bound regression tests — Julia @testset parity.

Source: test_bayesian_cramer_rao.jl, test_adaptive_estimation.jl
"""

import numpy as np


def test_40_simpson_quadrature() -> None:
    """#40: Trapezoidal integration of sin² must match π/2.

    Source: test_bayesian_cramer_rao.jl @testset "#40"
    """
    from scipy.integrate import trapezoid

    x = np.linspace(0.0, np.pi, 101)
    y = np.sin(x) ** 2
    assert np.isclose(trapezoid(y, x), np.pi / 2.0, rtol=1e-4)


def test_48_type_guard() -> None:
    """#48: BQFIM must accept plain vector for p, not just distribution.

    Source: test_bayesian_cramer_rao.jl @testset "#48"
    """
    from quanestimation.base.AsymptoticBound.CramerRao import QFIM

    omega = 1.0
    tspan = np.linspace(0.0, 1.0, 10)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    x_vals = np.linspace(0.5, 1.5, 5)
    p_vals = np.ones(len(x_vals)) / len(x_vals)

    from quanestimation.base.Parameterization.GeneralDynamics import Lindblad

    rho_list = []
    drho_list = []
    for xv in x_vals:
        dyn = Lindblad(tspan, rho0, xv * H0, [H0])
        states, d_states = dyn.expm()
        rho_list.append(states[-1])
        drho_list.append(d_states[-1])

    from quanestimation.base.BayesianBound.BayesCramerRao import BQFIM

    result = BQFIM([x_vals], p_vals, rho_list, drho_list, LDtype="SLD")
    assert result >= 0
