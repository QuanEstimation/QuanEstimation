import numpy as np
from scipy.linalg import sqrtm
from scipy.integrate import simps


def trace_norm(A, eps):
    if np.linalg.norm(A.conj().T - A) < eps:
        return np.sum(np.abs(np.linalg.eigvals(A)))
    else:
        return np.sum(np.linalg.svd(A, compute_uv=False))


def fidelity_dm(rho, sigma):
    rho_sqrtm = sqrtm(rho)
    fidelity_sqrt = np.trace(sqrtm(np.dot(rho_sqrtm, sigma, rho_sqrtm)))
    return np.real(fidelity_sqrt) ** 2


def fidelity_vec(psi, phi):
    overlap = np.dot(psi.conj().T, phi)
    return np.conj(overlap) * overlap


def helstrom_dm(rho, sigma, eps, P0=0.5):
    return np.real((1 - trace_norm(P0 * rho - (1 - P0) * sigma, eps)) / 2)


def helstrom_vec(psi, phi, n=1):
    return np.real((1 - np.sqrt(1 - fidelity_vec(psi, phi) ** n)) / 2)


def QZZB(x, p, rho, eps=1e-8):
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
        f_tp = simps(arr, x[0 : p_num - i])
        f_tau[i] = f_tp
    arr2 = [tau[m] * max(f_tau[m:]) for m in range(p_num)]
    I = simps(arr2, tau)
    return 0.5 * I
