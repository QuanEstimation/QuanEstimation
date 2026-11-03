import pytest
import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import (
    QFIM, 
    CFIM, 
    QFIM_Kraus, 
    QFIM_Bloch, 
    QFIM_Gauss, 
    LLD, 
    RLD, 
    FIM, 
    FI_Expt, 
    SLD
)
from ..utils import rand_rho, rand_drho


def test_CramerRao_SLD() -> None:
    """
    Test Cramer-Rao bound for parameterized quantum state.
    Checks QFIM and CFIM calculation for specific state and derivatives.
    """
    # Parameterized state
    theta = np.pi / 4
    phi = np.pi / 4
    rho = np.array([
        [np.cos(theta)**2, np.cos(theta) * np.sin(theta) * np.exp(-1j * phi)],
        [np.cos(theta) * np.sin(theta) * np.exp(1j * phi), np.sin(theta)**2]
    ])
    
    # State derivatives
    drho = [
        np.array([
            [-np.sin(2 * theta), np.cos(2 * theta) * np.exp(-1j * phi)],
            [np.cos(2 * theta) * np.exp(1j * phi), np.sin(2 * theta)]
        ]),
        np.array([
            [0, -1j * np.cos(theta) * np.sin(theta) * np.exp(-1j * phi)],
            [1j * np.cos(theta) * np.sin(theta) * np.exp(1j * phi), 0]
        ])
    ]
    
    # Calculate QFIM
    result = QFIM(rho, drho, LDtype="SLD")
    # Measurement operators
    M = [np.array([[1.0, 0.0], [0.0, 0.0]]), 
         np.array([[0.0, 0.0], [0.0, 1.0]])]
    # Calculate CFIM
    resultc = CFIM(rho, drho, M)
    
    # Verify results
    expected_qfim = np.array([[4.0, 0.0], [0.0, np.sin(2 * theta)**2]])
    expected_cfim = np.array([[4.0, 0.0], [0.0, 0.0]])
    assert np.allclose(result, expected_qfim)
    assert np.allclose(resultc, expected_cfim)

    # Test exportLD option
    result, SLD_res = QFIM(rho, drho, LDtype="SLD", exportLD=True)
    expected_SLD = SLD(rho, drho)
    assert np.allclose(SLD_res, expected_SLD)

    # Test invalid LDtype
    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="invalid")
    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="RLD")
    with pytest.raises(ValueError):
        QFIM(rho, drho, LDtype="LLD")


def test_CFIM_singleparameter() -> None:
    """
    Test Classical Fisher Information Matrix for single parameter.
    Checks CFIM calculation for specific state and derivative.
    """
    # Parameterized state
    theta = np.pi / 4
    rho = np.array([
        [np.cos(theta)**2, np.sin(theta) * np.cos(theta)],
        [np.sin(theta) * np.cos(theta), np.sin(theta)**2]
    ])
    
    # State derivative
    drho = [np.array([
        [-np.sin(2 * theta), np.cos(2 * theta)],
        [np.cos(2 * theta), np.sin(2 * theta)]
    ])]
    
    # Calculate CFIM
    result = CFIM(rho, drho, [])
    assert np.allclose(result, 2.0)


def test_QFIM_Kraus() -> None:
    """
    Test QFIM for Kraus representation.
    Checks QFIM calculation for specific Kraus operators.
    """
    # Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(0.5)]])
    K1 = np.array([[np.sqrt(0.5), 0], [0, 0]])
    K = [K0, K1]
    
    # Kraus operator derivatives
    dK = [
        [np.array([[0, 0], [0, -0.5 / np.sqrt(0.5)]])],
        [np.array([[0, 0.5 / np.sqrt(0.5)], [0, 0]])]
    ]
    
    # Probe state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # Calculate QFIM
    result = QFIM_Kraus(rho0, K, dK)
    assert np.allclose(result, 1.5)


def test_QFIM_Bloch() -> None:
    """
    Test QFIM for Bloch vector representation.
    Checks QFIM calculation for specific Bloch vector and derivatives.
    """
    # Bloch vector parameters
    theta = np.pi / 4
    phi = np.pi / 2
    eta = 0.8
    b = eta * np.array([
        np.sin(2 * theta) * np.cos(phi),
        np.sin(2 * theta) * np.sin(phi),
        np.cos(2 * theta)
    ])
    
    # Bloch vector derivatives
    db_theta = eta * np.array([
        2 * np.cos(2 * theta) * np.cos(phi),
        2 * np.cos(2 * theta) * np.sin(phi),
        -2 * np.sin(2 * theta)
    ])
    db_phi = eta * np.array([
        -np.sin(2 * theta) * np.sin(phi),
        np.sin(2 * theta) * np.cos(phi),
        0
    ])
    db = [db_theta, db_phi]
    
    # Calculate QFIM
    result = QFIM_Bloch(b, db)
    expected = np.array([
        [4.0 * eta**2, 0.0],
        [0.0, eta**2 * np.sin(2 * theta)**2]
    ])
    assert np.allclose(result, expected)


def test_QFIM_Bloch_pure() -> None:
    """
    Test QFIM for pure state in Bloch representation.
    Checks QFIM calculation for specific Bloch vector and derivatives.
    """
    # Bloch vector parameters
    theta = np.pi / 4
    phi = np.pi / 2
    eta = 1.0
    b = eta * np.array([
        np.sin(2 * theta) * np.cos(phi),
        np.sin(2 * theta) * np.sin(phi),
        np.cos(2 * theta)
    ])
    
    # Bloch vector derivatives
    db_theta = eta * np.array([
        2 * np.cos(2 * theta) * np.cos(phi),
        2 * np.cos(2 * theta) * np.sin(phi),
        -2 * np.sin(2 * theta)
    ])
    db_phi = eta * np.array([
        -np.sin(2 * theta) * np.sin(phi),
        np.sin(2 * theta) * np.cos(phi),
        0
    ])
    db = [db_theta, db_phi]
    
    # Calculate QFIM
    result = QFIM_Bloch(b, db)
    expected = np.array([
        [4.0 * eta**2, 0.0],
        [0.0, eta**2 * np.sin(2 * theta)**2]
    ])
    assert np.allclose(result, expected)


def test_QFIM_Bloch_highdimension() -> None:
    """
    Test QFIM for high-dimensional Bloch vector.
    Checks QFIM calculation for specific Bloch vector and derivative.
    """
    # Bloch vector parameters
    theta = np.pi / 4
    phi = np.pi / 2
    b = np.array([
        np.sin(2 * theta) * np.cos(phi),
        np.sin(2 * theta) * np.sin(phi),
        np.cos(2 * theta),
        0.0, 0.0, 0.0, 0.0, 0.0
    ])
    
    # Bloch vector derivative
    db = [np.array([
        2 * np.cos(2 * theta) * np.cos(phi),
        2 * np.cos(2 * theta) * np.sin(phi),
        -2 * np.sin(2 * theta),
        0.0, 0.0, 0.0, 0.0, 0.0
    ])]
    
    # Calculate QFIM
    result = QFIM_Bloch(b, db)
    assert np.allclose(result, 8.0)

    b_invalid = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0
    ])
    with pytest.raises(ValueError):
        QFIM_Bloch(b_invalid, db)


def test_QFIM_Gauss_multiparameter() -> None:
    """
    Test QFIM for Gaussian state with multiple parameters.
    Checks QFIM calculation for specific Gaussian state.
    """
    # Gaussian state parameters
    r = 0.8  # squeezing parameter
    beta = 0.5
    lamb = 1 / np.tanh(beta / 2)
    mu = np.array([0.0, 0.0])
    sigma = lamb * np.array([
        [np.cosh(2 * r), -np.sinh(2 * r)],
        [-np.sinh(2 * r), np.cosh(2 * r)]
    ])
    
    # Derivatives
    dmu = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    dlamb = -0.5 / (np.sinh(beta / 2)**2)
    dsigma = [
        dlamb * np.array([
            [np.cosh(2 * r), -np.sinh(2 * r)],
            [-np.sinh(2 * r), np.cosh(2 * r)]
        ]),
        lamb * 2 * np.array([
            [np.sinh(2 * r), -np.cosh(2 * r)],
            [-np.cosh(2 * r), np.sinh(2 * r)]
        ])
    ]
    
    # Calculate QFIM
    result = QFIM_Gauss(mu, dmu, sigma, dsigma)
    expected = np.array([
        [(lamb**2 - 1)**2 / (2 * (4 * lamb**2 - 1)), 0.0],
        [0.0, 8 * lamb**2 / (4 * lamb**2 + 1)]
    ])
    assert np.allclose(result, expected)


def test_QFIM_Gauss_singleparameter() -> None:
    """
    Test QFIM for Gaussian state with single parameter.
    Checks QFIM calculation for specific Gaussian state.
    """
    # Gaussian state parameters
    r = 0.8  # squeezing parameter
    beta = 0.5
    lamb = 1 / np.tanh(beta / 2)
    mu = np.array([0.0, 0.0])
    sigma = lamb * np.array([
        [np.cosh(2 * r), -np.sinh(2 * r)],
        [-np.sinh(2 * r), np.cosh(2 * r)]
    ])
    
    # Derivatives
    dmu = [np.array([0.0, 0.0])]
    dlamb = -0.5 / (np.sinh(beta / 2)**2)
    dsigma = [dlamb * np.array([
        [np.cosh(2 * r), -np.sinh(2 * r)],
        [-np.sinh(2 * r), np.cosh(2 * r)]
    ])]
    
    # Calculate QFIM
    result = QFIM_Gauss(mu, dmu, sigma, dsigma)
    expected = (lamb**2 - 1)**2 / (2 * (4 * lamb**2 - 1))
    assert np.allclose(result, expected)


def test_QFIM_LLD_singleparameter() -> None:
    """
    Test left logarithmic derivative for specific state.
    Checks LLD calculation and QFIM result.
    """
    # State parameters
    theta = np.pi / 4
    phi = np.pi / 4
    eta = 0.8
    rho = 0.5 * np.array([
        [1 + eta * np.cos(2 * theta), 
         eta * np.sin(2 * theta) * np.exp(-1j * phi)],
        [eta * np.sin(2 * theta) * np.exp(1j * phi), 
         1 - eta * np.cos(2 * theta)]
    ])
    
    # State derivative
    drho = [0.5 * np.array([
        [0.0, -1j * eta * np.sin(2 * theta) * np.exp(-1j * phi)],
        [1j * eta * np.sin(2 * theta) * np.exp(1j * phi), 0.0]
    ])]
    
    # Calculate LLD
    result = LLD(rho, drho, rep="original")
    expected = (1 / (1 - eta**2)) * np.array([
        [1j * eta**2 * np.sin(2 * theta)**2, 
         -1j * eta * (1 + eta * np.cos(2 * theta)) * np.sin(2 * theta) * np.exp(-1j * phi)],
        [1j * eta * (1 - eta * np.cos(2 * theta)) * np.sin(2 * theta) * np.exp(1j * phi), 
         -1j * eta**2 * np.sin(2 * theta)**2]
    ])
    assert np.allclose(result, expected)

    # Calculate QFIM
    result_QFIM = QFIM(rho, drho, LDtype="LLD")
    expected_QFIM = eta**2 * np.sin(2 * theta)**2 / (1.0 - eta**2)
    assert np.allclose(result_QFIM, expected_QFIM)

    # Test eigen representation
    result_eigen = LLD(rho, drho, rep="eigen")
    _, vec = np.linalg.eig(rho)
    expected_eigen = vec.conj().transpose() @ result @ vec
    assert np.allclose(result_eigen, expected_eigen)

    # Test invalid representation
    with pytest.raises(ValueError):
        LLD(rho, drho, rep="invalid")


def test_QFIM_RLD_singleparameter() -> None:
    """
    Test right logarithmic derivative for specific state.
    Checks RLD calculation and QFIM result.
    """
    # State parameters
    theta = np.pi / 4
    phi = np.pi / 4
    eta = 0.8
    rho = 0.5 * np.array([
        [1 + eta * np.cos(2 * theta), 
         eta * np.sin(2 * theta) * np.exp(-1j * phi)],
        [eta * np.sin(2 * theta) * np.exp(1j * phi), 
         1 - eta * np.cos(2 * theta)]
    ])
    
    # State derivative
    drho = [0.5 * np.array([
        [0.0, -1j * eta * np.sin(2 * theta) * np.exp(-1j * phi)],
        [1j * eta * np.sin(2 * theta) * np.exp(1j * phi), 0.0]
    ])]
    
    # Calculate RLD
    result = RLD(rho, drho, rep="original")
    expected = (1 / (1 - eta**2)) * np.array([
        [-1j * eta**2 * np.sin(2 * theta)**2, 
         -1j * eta * (1 - eta * np.cos(2 * theta)) * np.exp(-1j * phi) * np.sin(2 * theta)],
        [1j * eta * (1 + np.cos(2 * theta)) * np.exp(1j * phi) * np.sin(2 * theta), 
         1j * eta**2 * np.sin(2 * theta)**2]
    ])
    assert np.allclose(result, expected)

    # Calculate QFIM
    result_QFIM = QFIM(rho, drho, LDtype="RLD")
    expected_QFIM = eta**2 * np.sin(2 * theta)**2 / (1.0 - eta**2)
    assert np.allclose(result_QFIM, expected_QFIM)

    # Test eigen representation
    result_eigen = RLD(rho, drho, rep="eigen")
    _, vec = np.linalg.eig(rho)
    expected_eigen = vec.conj().transpose() @ result @ vec
    assert np.allclose(result_eigen, expected_eigen)

    # Test invalid representation
    with pytest.raises(ValueError):
        RLD(rho, drho, rep="invalid")


def test_FIM_singleparameter() -> None:
    """
    Test Fisher Information Matrix (FIM) for single parameter.
    Checks FIM calculation for classical probability distribution.
    """
    x = 1.0
    theta = np.pi / 3
    p = np.array([
        np.cos(x * theta)**2, 
        np.sin(x * theta)**2
    ])
    dp = [np.array([
        -x * np.sin(2 * x * theta), 
        x * np.sin(2 * x * theta)
    ])]
    result = FIM(p, dp)
    assert np.allclose(result, 4.0)

def test_FIM_multiparameter() -> None:
    """
    Test Fisher Information Matrix (FIM) for multiple parameters.
    Checks FIM calculation for classical probability distribution.
    """
    x = 1.0
    theta = np.pi / 3
    p = np.array([
        np.cos(x * theta)**2, 
        np.sin(x * theta)**2
    ])
    dp = [
        np.array([
            -theta * np.sin(2 * x * theta), 
            theta * np.sin(2 * x * theta)
        ]),
        np.array([
            -x * np.sin(2 * x * theta), 
            x * np.sin(2 * x * theta)
        ])
    ]
    result = FIM(p, dp)
    expected = np.array([
        [4.38649084, 4.1887902],
        [4.1887902, 4.0]
    ])
    assert np.allclose(result, expected)

def test_FI_Expt() -> None:
    """
    Test Fisher Information calculation for experimental data.
    Checks FI_Expt with different distribution types.
    """
    # Normal, Gamma, and Rayleigh distributions
    dx = 0.001
    y1_norm = np.random.normal(loc=0.0, scale=1.0, size=1000)
    y2_norm = np.random.normal(loc=dx, scale=1.0, size=1000)
    
    result_norm = FI_Expt(y1_norm, y2_norm, dx, ftype="norm")
    result_gamma = FI_Expt(y1_norm, y2_norm, dx, ftype="gamma")
    result_rayleigh = FI_Expt(y1_norm, y2_norm, dx, ftype="rayleigh")

    # Poisson distribution
    dx_poisson = 0.001
    y1_poi = np.random.poisson(lam=1, size=1000)
    y2_poi = np.random.poisson(lam=1 + dx_poisson, size=1000)
    result_poisson = FI_Expt(y1_poi, y2_poi, dx_poisson, ftype="poisson")

    # Verify results are floats
    assert all(isinstance(res, float) for res in [
        result_norm, result_gamma, result_rayleigh, result_poisson
    ])

    # Test invalid distribution type
    with pytest.raises(ValueError):
        FI_Expt(y1_norm, y2_norm, dx, ftype="invalid")

def test_invalid_input() -> None:
    """
    Test input validation for functions in the Cramer-Rao module.
    Verifies appropriate errors are raised for invalid inputs.
    """
    # Test CFIM with invalid inputs
    with pytest.raises(TypeError):
        CFIM(
            np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 1]]),
            None
        )

    # Test QFIM with invalid LDtype
    with pytest.raises(ValueError):
        QFIM(
            np.array([[1, 0], [0, 1]]),
            [np.array([[1, 0], [0, 1]])],
            LDtype="invalid"
        )

    # Test QFIM with invalid drho
    with pytest.raises(TypeError):
        QFIM(np.array([[1, 0], [0, 1]]), None)

    # Test QFIM_Bloch with invalid db
    with pytest.raises(TypeError):
        QFIM_Bloch(np.array([[1.0], [0.0], [1.0]]), None)

    # Test logarithmic derivatives with invalid drho
    with pytest.raises(TypeError):
        SLD(np.array([[1, 0], [0, 1]]), None)

    with pytest.raises(TypeError):
        LLD(np.array([[1, 0], [0, 1]]), None)

    with pytest.raises(TypeError):
        RLD(np.array([[1, 0], [0, 1]]), None)


def test_38_hermiticity_enforcement() -> None:
    """#38: Verify QFIM_SLD is real even with anti-Hermitian perturbation on rho."""
    for _ in range(5):
        N = 2
        rho = rand_rho(N)
        drho = rand_drho(N)
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        A_anti = A - A.conj().T
        eps = np.random.rand() * 1e-12
        rho_pert = rho + eps * A_anti
        F = QFIM(rho_pert, [drho], LDtype="SLD")
        assert abs(np.imag(F)) < 1e-12


def test_39_sld_hermiticity() -> None:
    """#39: Verify SLD operator is Hermitian for random matrices."""
    for _ in range(5):
        N = 2
        rho = rand_rho(N)
        drho = rand_drho(N)
        L = SLD(rho, [drho], rep="original")
        if isinstance(L, list):
            L = L[0]
        assert np.linalg.norm(L - L.conj().T) < 5e-13

        L_eig = SLD(rho, [drho], rep="eigen")
        if isinstance(L_eig, list):
            L_eig = L_eig[0]
        assert np.linalg.norm(L_eig - L_eig.conj().T) < 5e-13


# --- Phase 4.11 regression tests (Julia @testset parity) ---


def test_56_near_zero_eigenvalue() -> None:
    """#56: Near-zero eigenvalue (1e-16) must not cause NaN in QFIM.
    
    Source: test_fisher_information_matrix.jl @testset "#56"
    """
    rho = np.array([[1.0, 0.0], [0.0, 1e-16]], dtype=np.complex128)
    rho = (rho + rho.conj().T) / 2.0
    rho = rho / np.trace(rho)
    drho = [np.zeros((2, 2), dtype=np.complex128)]
    F = QFIM(rho, drho, LDtype="SLD")
    assert np.all(np.isfinite(F))
    assert not np.any(np.isnan(F))


def test_pinv_bypass() -> None:
    """pinv bypass: QFIM RLD/LLD must use operator functions, not raw pinv.

    Source: test_cramer_rao_bounds.jl @testset "pinv bypass"
    """
    # Full-rank: RLD ≈ LLD
    rho_full = np.array([[0.6, 0.0], [0.0, 0.4]], dtype=np.complex128)
    drho_full = [np.array([[0.1, 0.05], [0.05, -0.1]], dtype=np.complex128)]
    F_rld = QFIM(rho_full, drho_full, LDtype="RLD")
    F_lld = QFIM(rho_full, drho_full, LDtype="LLD")
    assert np.isclose(F_rld, F_lld, rtol=1e-10)
    # Singular: must raise (support of drho not in support of rho)
    rho_sing = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    drho_cross = [np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)]
    with pytest.raises(Exception):
        QFIM(rho_sing, drho_cross, LDtype="RLD")
    with pytest.raises(Exception):
        QFIM(rho_sing, drho_cross, LDtype="LLD")


@pytest.mark.parametrize("LDtype", ["SLD", "RLD", "LLD"])
def test_8_multiparam_qfim_dimensions(LDtype: str) -> None:
    """#8: QFIM must return correct dimensions for multi-parameter case.

    Source: test_fisher_information_matrix.jl @testset "#8"
    """
    rho = rand_rho(2)
    drho = [rand_drho(2), rand_drho(2), rand_drho(2)]
    F = QFIM(rho, drho, LDtype=LDtype)
    assert F.shape == (3, 3)
    # Pure state variant
    from quanestimation.base.AsymptoticBound.CramerRao import QFIM_pure
    Fp = QFIM_pure(rho, drho)
    assert Fp.shape == (3, 3)
