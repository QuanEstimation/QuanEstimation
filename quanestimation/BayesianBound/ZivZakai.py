import numpy as np
from scipy.linalg import sqrtm
from scipy.integrate import simpson


def trace_norm(A, eps):
    if np.linalg.norm(A.conj().T - A) < eps:
        return np.sum(np.abs(np.linalg.eigvals(A)))
    else:
        return np.sum(np.linalg.svd(A, compute_uv=False))

def helstrom_dm(rho, sigma, eps, P0=0.5):
    return np.real((1 - trace_norm(P0 * rho - (1 - P0) * sigma, eps)) / 2)

# def helstrom_vec(psi, phi, n=1):
#     return np.real((1 - np.sqrt(1 - fidelity(psi, phi) ** n)) / 2)


def QZZB(x, p, rho, eps=1e-8):
    r"""
    Calculation of the quantum Ziv-Zakai bound (QZZB). The expression of QZZB with a 
    prior distribution p(x) in a finite regime $[\alpha,\beta]$ is

    \begin{aligned}
        \mathrm{var}(\hat{x},\{\Pi_y\}) \geq &  \frac{1}{2}\int_0^\infty \mathrm{d}\tau\tau
        \mathcal{V}\int_{-\infty}^{\infty} \mathrm{d}x\min\!\left\{p(x), p(x+\tau)\right\} \nonumber \\
        & \times\left(1-\frac{1}{2}||\rho(x)-\rho(x+\tau)||\right).
    \end{aligned}

    Symbols:
        - $||\cdot||$: the trace norm
        - $\mathcal{V}$: the "valley-filling" operator satisfying $\mathcal{V}f(\tau)=\max_{h\geq 0}f(\tau+h)$. 
        - $\rho(x)$: the parameterized density matrix.

    Args:
        x (list): 
            The regimes of the parameters for the integral.
        p (np.ndarray): 
            The prior distribution as a multidimensional array.
        rho (list): 
            Parameterized density matrix as a multidimensional list.
        eps (float, optional): 
            Machine epsilon. Defaults to 1e-8.

    Returns:
        (float): 
            Quantum Ziv-Zakai bound (QZZB).

    Raises:
        ValueError: 
            If the length of x and p do not match.
    """

    if type(x[0]) == list or type(x[0]) == np.ndarray:
        x = x[0]
    p_num = len(p)
    tau = [xi - x[0] for xi in x]
    f_tau = np.zeros(p_num)
    for i in range(p_num):
        arr = [
            np.real(2 * min(p[j], p[j + i]) * helstrom_dm(rho[j], rho[j + i], eps))
            for j in range(p_num - i)
        ]
        f_tp = simpson(arr, x[0 : p_num - i])
        f_tau[i] = f_tp
    arr2 = [tau[m] * max(f_tau[m:]) for m in range(p_num)]
    I = simpson(arr2, tau)
    return 0.5 * I
