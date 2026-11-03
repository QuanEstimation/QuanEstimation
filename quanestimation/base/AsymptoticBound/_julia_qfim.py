"""Julia-backed Fisher-information kernels.

The functions here keep the historical Python signatures but delegate the
numerical work to :mod:`QuanEstimationBase` through ``juliacall``.  This is the
first step of the Julia-first thin-shell migration: the Python side owns data
conversion and argument validation, while Julia owns the numerical policy.
"""

import numpy as np
from juliacall import JuliaError
from juliacall import Main as _jl
from juliacall import convert as _jlconvert

from quanestimation.base.Common.Common import load_julia


def _call(function, *args, **kwargs):
    """Call a Julia function and translate argument errors to Python errors."""
    try:
        return function(*args, **kwargs)
    except JuliaError as exc:
        message = str(exc)
        if any(
            name in message
            for name in ("ArgumentError", "BoundsError", "DomainError", "DimensionMismatch")
        ):
            raise ValueError(message) from exc
        raise


def _matrix(value):
    return _jlconvert(
        _jl.Matrix[_jl.ComplexF64], np.asarray(value, dtype=np.complex128)
    )


def _matrices(values):
    return _jlconvert(
        _jl.Vector[_jl.Matrix[_jl.ComplexF64]],
        [np.asarray(value, dtype=np.complex128) for value in values],
    )


def _vector(value):
    return _jlconvert(_jl.Vector[_jl.Float64], np.asarray(value, dtype=np.float64))


def _vectors(values):
    return _jlconvert(
        _jl.Vector[_jl.Vector[_jl.Float64]],
        [np.asarray(value, dtype=np.float64) for value in values],
    )


def _precision(eps):
    """Translate the historical ``eps`` keyword into the Julia policy.

    ``eps`` is interpreted as a relative spectral floor, matching the
    deprecated Julia keyword. Passing ``eps=None`` (the default) uses the
    Julia 0.5 relative precision policy.
    """
    return {} if eps is None else {"guard": float(eps)}


def _qfim_value(value, num_params, real_only=True):
    array = np.asarray(value)
    if real_only:
        array = np.real(array)
    if num_params == 1:
        return array[0, 0]
    return array


def CFIM(rho, drho, M=None, eps=None):
    """Classical Fisher information (matrix), delegated to Julia.

    The arguments follow the historical Python API.  ``drho`` is a list of
    derivative matrices; ``M`` is an optional list of POVM elements.  The
    default ``eps=None`` uses the Julia 0.5 relative precision policy; a
    numeric ``eps`` is mapped to the relative spectral floor ``guard``.
    """
    if drho is None:
        raise TypeError("drho must not be None")
    if not isinstance(drho, (list, tuple)):
        raise TypeError("Please make sure drho is a list!")
    QJL = load_julia()
    kwargs = _precision(eps)
    rho_jl = _matrix(rho)
    drho_jl = _matrices(drho)
    if M is None or (isinstance(M, (list, tuple)) and len(M) == 0):
        value = _call(QJL.CFIM, rho_jl, drho_jl, **kwargs)
    else:
        value = _call(QJL.CFIM, rho_jl, drho_jl, _matrices(M), **kwargs)
    array = np.asarray(value)
    if len(drho) == 1:
        return float(np.real(array[0, 0]))
    return np.real(array)


def FIM(p, dp, eps=None):
    """Classical Fisher information from probabilities, delegated to Julia."""
    if dp is None:
        raise TypeError("dp must not be None")
    if not isinstance(dp, (list, tuple)):
        dp = [dp]
    QJL = load_julia()
    kwargs = _precision(eps)
    p_jl = _vector(p)
    if len(dp) == 1:
        value = _call(QJL.FIM, p_jl, _vector(dp[0]), **kwargs)
        return float(np.real(np.asarray(value)))
    dp_t = [[float(dp[a][j]) for a in range(len(dp))] for j in range(len(p))]
    value = _call(QJL.FIM, p_jl, _vectors(dp_t), **kwargs)
    return np.real(np.asarray(value))


def _logarithmic_derivative(name, rho, drho, rep, eps):
    if drho is None:
        raise TypeError("drho must not be None")
    if rep not in ("original", "eigen"):
        raise ValueError("The rep should be chosen in {'original', 'eigen'}.")
    QJL = load_julia()
    kwargs = _precision(eps)
    rho_jl = _matrix(rho)
    was_list = isinstance(drho, (list, tuple))
    drho_jl = _matrices(drho) if was_list else _matrix(drho)
    # Julia's eigenbasis ordering/phases differ from numpy's; compute the
    # operator in the original basis and transform with numpy's eigenvectors
    # to preserve the historical Python return convention.
    julia_rep = "original" if rep == "eigen" else rep
    value = _call(getattr(QJL, name), rho_jl, drho_jl, rep=julia_rep, **kwargs)

    def transform(item):
        if rep != "eigen":
            return np.asarray(item)
        eigvecs = np.linalg.eig(np.asarray(rho, dtype=np.complex128))[1]
        return eigvecs.conj().T @ np.asarray(item) @ eigvecs

    if not was_list:
        return transform(value)
    if len(drho) == 1:
        return transform(value[0])
    return [transform(item) for item in value]


def SLD(rho, drho, rep="original", eps=None):
    """Symmetric logarithmic derivative, delegated to Julia."""
    return _logarithmic_derivative("SLD", rho, drho, rep, eps)


def RLD(rho, drho, rep="original", eps=None):
    """Right logarithmic derivative, delegated to Julia."""
    return _logarithmic_derivative("RLD", rho, drho, rep, eps)


def LLD(rho, drho, rep="original", eps=None):
    """Left logarithmic derivative, delegated to Julia."""
    return _logarithmic_derivative("LLD", rho, drho, rep, eps)


def QFIM(rho, drho, LDtype="SLD", exportLD=False, eps=None):
    """Quantum Fisher information (matrix), delegated to Julia.

    ``LDtype`` is one of ``"SLD"``, ``"RLD"`` or ``"LLD"``.  With
    ``exportLD=True`` the logarithmic derivatives are returned alongside the
    QFIM, matching the historical Python behaviour.
    """
    if drho is None:
        raise TypeError("drho must not be None")
    if not isinstance(drho, (list, tuple)):
        drho = [drho]
    ldtype = str(LDtype).upper()
    if ldtype not in ("SLD", "RLD", "LLD"):
        raise ValueError("The LDtype should be chosen in {'SLD', 'RLD', 'LLD'}.")
    QJL = load_julia()
    kwargs = _precision(eps)
    value = _call(
        QJL.QFIM,
        _matrix(rho),
        _matrices(drho),
        LDtype=_jl.Symbol(ldtype),
        exportLD=bool(exportLD),
        **kwargs,
    )
    real_only = ldtype == "SLD"
    if exportLD:
        fisher, derivatives = value
        fisher_out = _qfim_value(fisher, len(drho), real_only=real_only)
        if len(drho) == 1:
            derivatives_out = np.asarray(derivatives[0])
        else:
            derivatives_out = [np.asarray(item) for item in derivatives]
        return fisher_out, derivatives_out
    return _qfim_value(value, len(drho), real_only=real_only)
