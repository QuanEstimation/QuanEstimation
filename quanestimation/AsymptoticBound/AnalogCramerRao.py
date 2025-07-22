import numpy as np
import scipy as sp
import cvxpy as cp
from quanestimation.Common.Common import suN_generator
from quanestimation.AsymptoticBound.CramerRao import QFIM
from numpy.linalg import matrix_rank


def HCRB(rho, drho, W, eps=1e-8):
    r"""
    Calculate the Holevo Cramer-Rao bound (HCRB) via semidefinite programming (SDP).

    The HCRB is defined as:

    $$
    \min_{\{X_i\}} \left\{ \mathrm{Tr}(\mathrm{Re}Z) + \mathrm{Tr}(| \mathrm{Im} Z |) \right\}, 
    $$

    where $Z_{ij} = \mathrm{Tr}(\rho X_i X_j)$ and $V$ is the covariance matrix.

    Args:
        rho (np.array): 
            Density matrix.
        drho (list): 
            Derivatives of the density matrix with respect to unknown parameters.  
            For example, `drho[0]` is the derivative with respect to the first parameter.
        W (np.array): 
            Weight matrix for the bound.
        eps (float, optional): 
            Machine epsilon for numerical stability.

    Returns: 
        (float): 
            The value of the Holevo Cramer-Rao bound.

    Raises:
        TypeError: If `drho` is not a list.

    Notes:
        In the single-parameter scenario, the HCRB is equivalent to the QFI.  
        For a rank-one weight matrix, the HCRB is equivalent to the inverse of the QFIM.
    """

    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    if len(drho) == 1:
        print(
            "In single parameter scenario, HCRB is equivalent to QFI. "
            "Returning QFI value."
        )
        return QFIM(rho, drho, eps=eps)
    
    if matrix_rank(W) == 1:
        print(
            "For rank-one weight matrix, HCRB is equivalent to QFIM. "
            "Returning Tr(W @ inv(QFIM))."
        )
        qfim = QFIM(rho, drho, eps=eps)
        return np.trace(W @ np.linalg.pinv(qfim))
    dim = len(rho)
    num = dim * dim
    num_params = len(drho)

    # Generate basis matrices
    basis = [np.identity(dim)] + suN_generator(dim)
    basis = [b / np.sqrt(2) for b in basis]

    # Compute vectorized derivatives
    vec_drho = []
    for param_idx in range(num_params):
        components = [
            np.real(np.trace(drho[param_idx] @ basis_mat))
            for basis_mat in basis
        ]
        vec_drho.append(np.array(components))

    # Compute S matrix
    S = np.zeros((num, num), dtype=np.complex128)
    for i in range(num):
        for j in range(num):
            S[i, j] = np.trace(basis[i] @ basis[j] @ rho)

    # Regularize and factor S
    precision = len(str(int(1 / eps))) - 1
    lu, d, _ = sp.linalg.ldl(S.round(precision))
    R = (lu @ sp.linalg.sqrtm(d)).conj().T

    # Define optimization variables
    V = cp.Variable((num_params, num_params))
    X = cp.Variable((num, num_params))

    # Define constraints
    constraints = [
        cp.bmat([
            [V, X.T @ R.conj().T],
            [R @ X, np.identity(num)]
        ]) >> 0
    ]

    # Add linear constraints
    for i in range(num_params):
        for j in range(num_params):
            constraint_val = X[:, i].T @ vec_drho[j]
            if i == j:
                constraints.append(constraint_val == 1)
            else:
                constraints.append(constraint_val == 0)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cp.trace(W @ V)), constraints)
    problem.solve()

    return problem.value

def NHB(rho, drho, W):
    r"""
    Calculation of the Nagaoka-Hayashi bound (NHB) via the semidefinite program (SDP).

    The NHB is defined as:

    $$
    \min_{X} \left\{ \mathrm{Tr}[W \mathrm{Re}(Z)] + \|\sqrt{W} \mathrm{Im}(Z) \sqrt{W}\|_1 \right\}, 
    $$
    
    where $Z_{ij} = \mathrm{Tr}(\rho X_i X_j)$ and $V$ is the covariance matrix.

    Args:
        rho (np.array): 
            Density matrix.
        drho (list): 
            Derivatives of the density matrix with respect to unknown parameters.  
            For example, `drho[0]` is the derivative with respect to the first parameter.
        W (np.array): 
            Weight matrix for the bound.

    Returns: 
        (float): 
            The value of the Nagaoka-Hayashi bound.

    Raises:
        TypeError: 
            If `drho` is not a list.
    """
    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")
    
    dim = len(rho)
    num_params = len(drho)
    
    # Initialize a temporary matrix for L components
    L_temp = [[None for _ in range(num_params)] for _ in range(num_params)]
    
    # Create Hermitian variables for the upper triangle and mirror to lower triangle
    for i in range(num_params):
        for j in range(i, num_params):
            L_temp[i][j] = cp.Variable((dim, dim), hermitian=True)
            if i != j:
                L_temp[j][i] = L_temp[i][j]
    
    # Construct the block matrix L
    L_blocks = [cp.hstack(L_temp[i]) for i in range(num_params)]
    L = cp.vstack(L_blocks)
    
    # Create Hermitian variables for X
    X = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_params)]
    
    # Construct the block matrix constraint
    block_matrix = cp.bmat([
        [L, cp.vstack(X)],
        [cp.hstack(X), np.identity(dim)]
    ])
    constraints = [block_matrix >> 0]
    
    # Add trace constraints
    for i in range(num_params):
        constraints.append(cp.trace(X[i] @ rho) == 0)
        for j in range(num_params):
            if i == j:
                constraints.append(cp.trace(X[i] @ drho[j]) == 1)
            else:
                constraints.append(cp.trace(X[i] @ drho[j]) == 0)
    
    # Define and solve the optimization problem
    objective = cp.Minimize(cp.real(cp.trace(cp.kron(W, rho) @ L)))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value
