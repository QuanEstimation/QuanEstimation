"""Regression tests for the Julia-backed Fisher-information kernels."""

import numpy as np

from quanestimation.base.AsymptoticBound.CramerRao import CFIM, QFIM


def test_low_temperature_thermometry() -> None:
    """The delegated QFIM/CFIM reproduce the analytic low-temperature QFI.

    This is the numerical-stability benchmark of QuanEstimation 0.5: the old
    Python ``eps=1e-8`` absolute threshold silently dropped the excited-state
    population at ``T/epsilon = 0.05``.
    """
    epsilon, temperature = 1.0, 0.05
    population = 1.0 / (1.0 + np.exp(epsilon / temperature))
    rho = np.array(
        [[1.0 - population, 0.0], [0.0, population]], dtype=np.complex128
    )
    d_population = (
        population * (1.0 - population) * epsilon / temperature**2
    )
    drho = np.array(
        [[-d_population, 0.0], [0.0, d_population]], dtype=np.complex128
    )
    measurement = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    ]
    analytic = population * (1.0 - population) * (epsilon / temperature**2) ** 2

    assert np.isclose(QFIM(rho, [drho]), analytic, rtol=1e-12, atol=0.0)
    assert np.isclose(
        CFIM(rho, [drho], measurement), analytic, rtol=1e-12, atol=0.0
    )
