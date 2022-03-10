from functools import wraps
from scipy.linalg import sqrtm
from numpy.linalg import eigvals
import numpy as np
from more_itertools import zip_broadcast


def SpinSqueezing(rho, basis="Dicke", output="KU"):
    """
    Description: Calculate spin squeezing parameter for a density matrix.

    ---------
    Inputs
    ---------
    N :
       --description: particle number.
       --type: int
    rho :
       --description: density matrix.
       --type: matrix

    output:
       --description: if output=='KU',the output of the squeezing parameter is defined by
                      Kitagawa and Ueda. if output=='WBIMH',the output of the squeezing
                      parameter is defined by Wineland et al.
       --type: string {'KU', 'WBIMH'}

    ----------
    Returns
    ----------
    Xi :
       --description: squeezing_parameter.
       --type: float

    """
    N = len(rho) - 1

    coef = 4.0 / float(N)
    j = N / 2
    offdiag = [
        np.sqrt(float(j * (j + 1) - m * (m + 1))) for m in np.arange(j, -j - 1, -1)
    ][1:]
    Jp = np.matrix(np.diag(offdiag, 1))
    Jx = 0.5 * (Jp + Jp.H)
    Jy = -0.5 * 1j * (Jp - Jp.H)
    Jz = np.diag(np.arange(j, -j - 1, -1))
    Jx_mean = np.trace(rho * Jx)
    Jy_mean = np.trace(rho * Jy)
    Jz_mean = np.trace(rho * Jz)

    costheta = Jz_mean / np.sqrt(Jx_mean**2 + Jy_mean**2 + Jz_mean**2)
    sintheta = np.sin(np.arccos(costheta))
    cosphi = Jx_mean / np.sqrt(Jx_mean**2 + Jy_mean**2)
    if Jy_mean > 0:
        sinphi = np.sin(np.arccos(cosphi))
    else:
        sinphi = np.sin(2 * np.pi - np.arccos(cosphi))
    Jn1 = -Jx * sinphi + Jy * cosphi
    Jn2 = -Jx * costheta * cosphi - Jy * costheta * sinphi + Jz * sintheta
    A = np.trace(rho * (Jn1 * Jn1 - Jn2 * Jn2))
    B = np.trace(rho * (Jn1 * Jn2 + Jn2 * Jn1))
    C = np.trace(rho * (Jn1 * Jn1 + Jn2 * Jn2))

    V_minus = 0.5 * (C - np.sqrt(A**2 + B**2))
    V_minus = np.real(V_minus)
    Xi = coef * V_minus
    if Xi > 1.0:
        Xi = 1.0

    if output == "KU":
        Xi = Xi
    elif output == "WBIMH":
        Xi = (N / 2) ** 2 * Xi / (Jx_mean**2 + Jy_mean**2 + Jz_mean**2)
    else:
        raise NameError("NameError: output should be choosen in {KU, WBIMH}")

    return Xi


def TargetTime(f, tspan, func, *args, **kwargs):
    args = list(zip_broadcast(*args))

    f_last = func(*(args[0]), **kwargs)
    idx = 1
    f_now = func(*(args[1]), **kwargs)

    while (f_now - f) * (f_last - f) > 0 and idx < (len(tspan) - 1):
        f_last = f_now
        idx += 1
        f_now = func(*(args[idx]), **kwargs)

    return tspan[idx]
