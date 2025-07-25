import numpy as np

def SpinSqueezing(rho, basis="Dicke", output="KU"):
    r"""
    Calculation of the spin squeezing parameter for a density matrix.

    The spin squeezing parameter $\xi$ given by Kitagawa and Ueda is defined as:

    $$
    \xi^2 = \frac{N(\Delta J_{\vec{n}_1})^2}{\langle J_{\vec{n}_3}\rangle^2}
    $$

    where $J_{\vec{n}_i}$ are the collective spin operators.

    The spin squeezing parameter $\xi$ given by Wineland etal. is defined as:

    $$
    \xi^2 = \left(\frac{j}{\langle \vec{J}\rangle}\right)^2 \frac{N(\Delta J_{\vec{n}_1})^2}{\langle J_{\vec{n}_3}\rangle^2}
    $$

    Args:
        rho (np.array): 
            Density matrix.
        basis (str, optional): 
            Basis to use: "Dicke" (default) or "Pauli".
        output (str, optional): 
            Type of spin squeezing to calculate:  
                - "KU": Kitagawa-Ueda squeezing parameter.  
                - "WBIMH": Wineland et al. squeezing parameter.  

    Returns:
        (float): 
            Spin squeezing parameter.

    Raises:
        ValueError: If `basis` has invalid value.  
        ValueError: If `output` has invalid value.  
    """

    if basis == "Pauli":
        N = int(np.log(len(rho)) / np.log(2))
        j = N / 2
        coef = 4.0 / float(N)
        sp = np.array([[0.0, 1.0], [0.0, 0.0]])
        sz = np.array([[1., 0.], [0., -1.]])
        jp = []
        jz = []
        for i in range(N):
            if i == 0:
                jp_tp = np.kron(sp, np.identity(2 ** (N - 1)))
                jz_tp = np.kron(sz, np.identity(2 ** (N - 1)))
            elif i == N - 1:
                jp_tp = np.kron(np.identity(2 ** (N - 1)), sp)
                jz_tp = np.kron(np.identity(2 ** (N - 1)), sz)
            else:
                jp_tp = np.kron(
                    np.identity(2 ** i), 
                    np.kron(sp, np.identity(2 ** (N - 1 - i)))
                )
                jz_tp = np.kron(
                    np.identity(2 ** i), 
                    np.kron(sz, np.identity(2 ** (N - 1 - i)))
                )
            jp.append(jp_tp)
            jz.append(jz_tp)
        Jp = sum(jp)
        Jz = 0.5 * sum(jz)
    elif basis == "Dicke":
        N = len(rho) - 1
        j = N / 2 
        coef = 4.0 / float(N)       
        offdiag = [
            np.sqrt(float(j * (j + 1) - m * (m + 1))) 
            for m in np.arange(j, -j - 1, -1)
        ][1:]
        # Ensure we create a complex array
        Jp = np.diag(offdiag, 1).astype(complex)
        Jz = np.diag(np.arange(j, -j - 1, -1))
    else:
        valid_types = ["Dicke", "Pauli"]
        raise ValueError(
                f"Invalid basis: '{basis}'. Supported types: {', '.join(valid_types)}"
            )    
    
    Jx = 0.5 * (Jp + np.conj(Jp).T)
    Jy = -0.5 * 1j * (Jp - np.conj(Jp).T)
    
    Jx_mean = np.trace(rho @ Jx)
    Jy_mean = np.trace(rho @ Jy)
    Jz_mean = np.trace(rho @ Jz)

    if Jx_mean == 0 and Jy_mean == 0:
        if Jz_mean == 0:
            raise ValueError("The density matrix does not have a valid spin squeezing.")
        else:
            A = np.trace(rho @ (Jx @ Jx - Jy @ Jy))
            B = np.trace(rho @ (Jx @ Jy + Jy @ Jx))
            C = np.trace(rho @ (Jx @ Jx + Jy @ Jy))
    else:
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

    if output == "KU":
        pass
    elif output == "WBIMH":
        xi = (N / 2)**2 * xi / (Jx_mean**2 + Jy_mean**2 + Jz_mean**2)
    else:
        valid_types = ["KU", "WBIMH"]
        raise ValueError(
                f"Invalid basis: '{basis}'. Supported types: {', '.join(valid_types)}"
            )  

    return xi


def TargetTime(f, tspan, func, *args, **kwargs):
    r"""
    Calculation of the time to reach a given precision limit. 

    This function finds the earliest time $t$ in `tspan` where the objective 
    function `func` reaches or crosses the target value $f$. The first argument 
    of func must be the time variable.

    Args:
        f (float): 
            The target value of the objective function.
        tspan (array): 
            Time points for the evolution.
        func (callable): 
            The objective function to evaluate. Must return a float.
        *args (tuple): 
            Positional arguments to pass to `func`.
        **kwargs (dict): 
            Keyword arguments to pass to `func`.

    Returns:
        (float): 
            Time to reach the given target precision.
    """
    # Check if we're already at the target at the first point
    f0 = func(tspan[0], *args, **kwargs)
    if np.isclose(f0, f, atol=1e-8):
        return tspan[0]
    
    # Iterate through time points
    for i in range(1, len(tspan)):
        f1 = func(tspan[i], *args, **kwargs)
        
        # Check if we've crossed the target
        if (f0 - f) * (f1 - f) <= 0:
            return tspan[i]
        elif np.isclose(f1, f, atol=1e-8):
            return tspan[i]
        
        f0 = f1
    
    # No crossing found
    print("No time is found in the given time span to reach the target.")

    return None
