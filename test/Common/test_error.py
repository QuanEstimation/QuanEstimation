"""Error evaluation and control regression tests.

Source: test_error.jl @testset "error_evaluation", "error_control"
"""

import numpy as np


def test_error_evaluation() -> None:
    """error_evaluation() must run without error.

    Source: test_error.jl @testset "error_evaluation"
    """
    from quanestimation.base.Common.Common import error_evaluation
    from quanestimation.base import QJL

    rho0 = np.ones((2, 2), dtype=np.complex128) * 0.5
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H0 = 0.5 * sz
    dH = [0.5 * sz]
    tspan = np.linspace(0.0, 2.0, 100)
    Hc = [sx, sz, sp]
    decay = [[sp, 0.0], [sm, 0.1]]
    ctrl = [np.zeros(len(tspan) - 1) for _ in Hc]

    dyn = QJL.Lindblad(H0, dH, tspan, Hc, decay, ctrl=ctrl, dyn_method="Expm")
    scheme = QJL.GeneralScheme(probe=rho0, param=dyn)
    error_evaluation(scheme)


def test_error_control() -> None:
    """error_control() must run without error.

    Source: test_error.jl @testset "error_control"
    """
    from quanestimation.base.Common.Common import error_control
    from quanestimation.base import QJL

    rho0 = np.ones((2, 2), dtype=np.complex128) * 0.5
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H0 = 0.5 * sz
    dH = [0.5 * sz]
    tspan = np.linspace(0.0, 2.0, 100)
    Hc = [sx, sz, sp]
    decay = [[sp, 0.0], [sm, 0.1]]
    ctrl = [np.zeros(len(tspan) - 1) for _ in Hc]

    dyn = QJL.Lindblad(H0, dH, tspan, Hc, decay, ctrl=ctrl, dyn_method="Expm")
    scheme = QJL.GeneralScheme(probe=rho0, param=dyn)
    error_control(scheme)
