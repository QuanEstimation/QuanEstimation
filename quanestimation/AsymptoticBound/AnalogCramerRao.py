import numpy as np
import scipy as sp
import cvxpy as cp
from quanestimation.Common.Common import suN_generator
from quanestimation.AsymptoticBound.CramerRao import QFIM
from numpy.linalg import matrix_rank


def HCRB(rho, drho, W, eps=1e-8):
    """
    Calculation of the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **W:** `matrix`
        -- Weight matrix.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **HCRB:** `float`
        -- The value of Holevo Cramer-Rao bound.
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    if len(drho) == 1:
        print(
            "In single parameter scenario, HCRB is equivalent to QFI. This function will return the value of QFI."
        )
        f = QFIM(rho, drho, eps=eps)
        return f
    elif matrix_rank(W) == 1:
        print(
            "For rank-one weight matrix, the HCRB is equivalent to QFIM. This function will return the value of Tr(WF^{-1})."
        )
        F = QFIM(rho, drho, eps=eps)
        return np.trace(np.dot(W, np.linalg.pinv(F)))
    else:
        dim = len(rho)
        num = dim * dim
        para_num = len(drho)

        Lambda = [np.identity(dim)] + suN_generator(dim)
        Lambda = Lambda / np.sqrt(2)

        vec_drho = [[] for i in range(para_num)]
        for pi in range(para_num):
            vec_drho[pi] = np.array(
                [
                    np.real(np.trace(np.dot(drho[pi], Lambda[i])))
                    for i in range(len(Lambda))
                ]
            )

        S = np.zeros((num, num), dtype=np.complex128)
        for a in range(num):
            for b in range(num):
                S[a][b] = np.trace(np.dot(Lambda[a], np.dot(Lambda[b], rho)))

        accu = len(str(int(1 / eps))) - 1
        lu, d, perm = sp.linalg.ldl(S.round(accu))
        R = np.dot(lu, sp.linalg.sqrtm(d)).conj().T
        # ============optimization variables================
        V = cp.Variable((para_num, para_num))
        X = cp.Variable((num, para_num))
        # ================add constraints===================
        constraints = [cp.bmat([[V, X.T @ R.conj().T], [R @ X, np.identity(num)]]) >> 0]
        for i in range(para_num):
            for j in range(para_num):
                if i == j:
                    constraints += [X[:, i].T @ vec_drho[j] == 1]
                else:
                    constraints += [X[:, i].T @ vec_drho[j] == 0]

        prob = cp.Problem(cp.Minimize(cp.trace(W @ V)), constraints)
        prob.solve()

        return prob.value

def NHB(rho, drho, W):
    """
    Calculation of the Nagaoka-Hayashi bound (NHB) via the semidefinite program (SDP).

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **W:** `matrix`
        -- Weight matrix.

    Returns
    ----------
    **NHB:** `float`
        -- The value of Nagaoka-Hayashi bound.
    """
    dim = len(rho)
    para_num = len(drho)
    
    L_tp = [[[] for i in range(para_num)] for j in range(para_num)]
    for para_i in range(para_num):
        for para_j in range(para_i, para_num):
            L_tp[para_i][para_j] = cp.Variable((dim, dim), hermitian=True)
            L_tp[para_j][para_i] = L_tp[para_i][para_j]
    L = cp.vstack([cp.hstack(L_tp[i]) for i in range(para_num)])
    X = [cp.Variable((dim, dim), hermitian=True) for j in range(para_num)]
    
    constraints = [cp.bmat([[L, cp.vstack(X)], [cp.hstack(X), np.identity(dim)]])  >> 0]
    
    for i in range(para_num):
        constraints += [cp.trace(X[i] @ rho) == 0]
        for j in range(para_num):
            if i == j:
                constraints += [cp.trace(X[i] @ drho[j]) == 1]
            else:
                constraints += [cp.trace(X[i] @ drho[j]) == 0]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(cp.kron(W, rho) @ L))), constraints)
    prob.solve()

    return prob.value
