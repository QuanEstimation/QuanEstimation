import numpy as np
from more_itertools import zip_broadcast


def SpinSqueezing(rho, basis="Dicke", output="KU"):
    r"""
    Calculation of the spin squeezing parameter for a density matrix.

    The spin squeezing parameter $\xi$ is defined as:
    \begin{align}
    \xi^2 = \frac{N(\Delta J_{\vec{n}_1})^2}{\langle J_{\vec{n}_3}\rangle^2}
    \end{align}
    where $J_{\vec{n}_i}$ are the collective spin operators.

    ## Parameters
    **rho** : matrix  
        Density matrix.

    **basis** : str, optional  
        Basis to use: "Dicke" (default) or "Pauli".

    **output** : str, optional  
        Type of spin squeezing to calculate:  
        - "KU": Kitagawa-Ueda squeezing parameter  
        - "WBIMH": Wineland et al. squeezing parameter  

    ## Returns
    **$\xi$** : float  
        Spin squeezing parameter.

    ## Raises
    **ValueError**  
        If basis has invalid value.  
    **NameError**  
        If output has invalid value.  
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
        # Ensure we create a complex array
        Jp = np.diag(offdiag, 1).astype(complex)
    else:
        valid_types = ["Dicke", "Pauli"]
        raise ValueError(
                f"Invalid basis: '{basis}'. Supported types: {', '.join(valid_types)}"
            )    
    
    Jx = 0.5 * (Jp + np.conj(Jp).T)
    Jy = -0.5 * 1j * (Jp - np.conj(Jp).T)
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
    r"""
    Calculation of the time to reach a given precision limit.

    This function finds the earliest time $t$ in `tspan` where the objective 
    function `func` reaches or crosses the target value $f$.

    ## Parameters
    **f** : float  
        The target value of the objective function.

    **tspan** : array  
        Time points for the evolution.

    **func** : callable  
        The objective function to evaluate. Must return a float.

    ***args**  
        Positional arguments to pass to `func`.

    ****kwargs**  
        Keyword arguments to pass to `func`.

    ## Returns
    float  
        Time to reach the given target precision.
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
