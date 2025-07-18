import numpy as np
from more_itertools import zip_broadcast


def SpinSqueezing(rho, basis="Dicke", output="KU"):
    r"""
    Calculate spin squeezing parameter for a density matrix.

    Parameters
    ----------
    rho : matrix
        Density matrix.
    basis : string, optional
        Basis to use: "Dicke" (default) or "Pauli".
    output : string, optional
        Type of spin squeezing to calculate: "KU" (default) or "WBIMH".

    Returns
    -------
    xi : float
        Spin squeezing parameter.

    Raises
    ------
    NameError
        If invalid output type is provided.
    """
    N = len(rho) - 1
    coef = 4.0 / float(N)
    j = N / 2
    
    if basis == "Pauli":
        sp = np.array([[0.0, 1.0], [0.0, 0.0]])
        jp = []
        for i in range(N):
            if i == 0:
                jp_tp = np.kron(sp, np.identity(2 ** (N - 1)))
            elif i == N - 1:
                jp_tp = np.kron(np.identity(2 ** (N - 1)), sp)
            else:
                jp_tp = np.kron(
                    np.identity(2 ** i), 
                    np.kron(sp, np.identity(2 ** (N - 1 - i)))
                )
            jp.append(jp_tp)
        Jp = sum(jp)
    elif basis == "Dicke":
        offdiag = [
            np.sqrt(float(j * (j + 1) - m * (m + 1))) 
            for m in np.arange(j, -j - 1, -1)
        ][1:]
        Jp = np.diag(offdiag, 1)
    else:
        valid_types = ["Dicke", "Pauli"]
        raise ValueError(
                f"Invalid basis: '{basis}'. Supported types: {', '.join(valid_types)}"
            )    
    
    Jx = 0.5 * (Jp + Jp.conj().T)
    Jy = -0.5 * 1j * (Jp - Jp.conj().T)
    Jz = np.diag(np.arange(j, -j - 1, -1))
    
    Jx_mean = np.trace(rho @ Jx)
    Jy_mean = np.trace(rho @ Jy)
    Jz_mean = np.trace(rho @ Jz)

    costheta = Jz_mean / np.sqrt(Jx_mean**2 + Jy_mean**2 + Jz_mean**2)
    sintheta = np.sin(np.arccos(costheta))
    cosphi = Jx_mean / np.sqrt(Jx_mean**2 + Jy_mean**2)
    sinphi = (np.sin(np.arccos(cosphi)) if Jy_mean > 0 
              else np.sin(2 * np.pi - np.arccos(cosphi)))
    
    Jn1 = -Jx * sinphi + Jy * cosphi
    Jn2 = (-Jx * costheta * cosphi 
           - Jy * costheta * sinphi 
           + Jz * sintheta)
    
    A = np.trace(rho @ (Jn1 @ Jn1 - Jn2 @ Jn2))
    B = np.trace(rho @ (Jn1 @ Jn2 + Jn2 @ Jn1))
    C = np.trace(rho @ (Jn1 @ Jn1 + Jn2 @ Jn2))
    
    V_minus = 0.5 * (C - np.sqrt(A**2 + B**2))
    V_minus = np.real(V_minus)
    xi = coef * V_minus
    xi = min(xi, 1.0)  # Cap at 1.0

    if output == "KU":
        pass
    elif output == "WBIMH":
        xi = (N / 2)**2 * xi / (Jx_mean**2 + Jy_mean**2 + Jz_mean**2)
    else:
        raise NameError("output should be either 'KU' or 'WBIMH'")

    return xi


def TargetTime(f, tspan, func, *args, **kwargs):
    """
    Calculation of the time to reach a given precision limit.

    Parameters
    ----------
    > **f:** `float`
        -- The given value of the objective function.

    > **tspan:** `array`
        -- Time length for the evolution.

    > **func:** `array`
        -- The function for calculating the objective function.

    > ***args:** `string`
        -- The corresponding input parameter.

    > ****kwargs:** `string`
        -- Keyword arguments in `func`.

    Returns
    ----------
    **time:** `float`
        -- Time to reach the given target.
    """

    args = list(zip_broadcast(*args))

    f_last = func(*(args[0]), **kwargs)
    idx = 1
    f_now = func(*(args[1]), **kwargs)

    while (f_now - f) * (f_last - f) > 0 and idx < (len(tspan) - 1):
        f_last = f_now
        idx += 1
        f_now = func(*(args[idx]), **kwargs)

    return tspan[idx]
