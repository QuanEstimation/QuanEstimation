"""Adaptive MZI regression test — Julia @testset parity.

Source: test_adaptive_estimation_MZI.jl @testset "Adaptive MZI"
"""

import numpy as np


def test_adaptive_mzi() -> None:
    """Adaptive MZI: DE and PSO offline must run without error.

    Source: test_adaptive_estimation_MZI.jl @testset "Adaptive MZI"
    """
    from quanestimation.base.AdaptiveScheme.Adapt_MZI import Adapt_MZI
    from quanestimation.base.Common.Common import basis

    N = 3
    dim = N + 1
    psi = np.zeros(dim * dim, dtype=np.complex128)
    for k in range(1, N + 2):
        v1 = basis(dim, k - 1).flatten()
        v2 = basis(dim, N - k + 1).flatten()
        psi += np.sin(k * np.pi / (N + 2)) * np.kron(v1, v2)
    psi = psi * np.sqrt(2.0 / (2.0 + N))
    rho0 = np.outer(psi, psi.conj())

    x = np.linspace(-np.pi, np.pi, 5)
    p = (1.0 / (x[-1] - x[0])) * np.ones(len(x))

    apt = Adapt_MZI(x, p, rho0)
    assert apt is not None

    apt.offline("sharpness", method="DE", p_num=10, seed=1234,
                max_episode=10, c=1.0, cr=0.5)
    apt.offline("sharpness", method="PSO", p_num=10, seed=1234,
                max_episode=10, c0=1.0, c1=2.0, c2=2.0)
