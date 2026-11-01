import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.base.Common.Common import SIC, suN_generator

from quanestimation.base.AsymptoticBound._julia_qfim import (
    CFIM,
    FIM,
    SLD,
    RLD,
    LLD,
    QFIM,
)
from scipy.integrate import quad
from scipy.stats import norm, poisson, rayleigh, gamma

def FI_Expt(y1, y2, dx, ftype="norm"):
    """
    Calculate the classical Fisher information (CFI) based on experimental data.

    Args:
        y1 (np.array): 
            Experimental data obtained at the true parameter value.
        y2 (np.array): 
            Experimental data obtained at parameter value shifted by dx.
        dx (float): 
            Small known parameter shift.
        ftype (str, optional): 
            Probability distribution of the data. Options:  
                - "norm": normal distribution (default).  
                - "gamma": gamma distribution.  
                - "rayleigh": Rayleigh distribution.  
                - "poisson": Poisson distribution.  

    Returns: 
        (float): 
            Classical Fisher information

    Raises:
        ValueError: 
            If `ftype` is not one of the supported types ("norm", "poisson", "gamma", "rayleigh").    

    Notes:
        The current implementation may be unstable and is subject to future modification.
    """
    fidelity = 0.0
    if ftype == "norm":
        mu_true, std_true = norm.fit(y1)
        mu_shifted, std_shifted = norm.fit(y2)
        f_function = lambda x: np.sqrt(
            norm.pdf(x, mu_true, std_true) * norm.pdf(x, mu_shifted, std_shifted)
        )
        fidelity, _ = quad(f_function, -np.inf, np.inf)
        
    elif ftype == "gamma":
        a_true, alpha_true, beta_true = gamma.fit(y1)
        a_shifted, alpha_shifted, beta_shifted = gamma.fit(y2)
        f_function = lambda x: np.sqrt(
            gamma.pdf(x, a_true, alpha_true, beta_true) *
            gamma.pdf(x, a_shifted, alpha_shifted, beta_shifted)
        )
        fidelity, _ = quad(f_function, 0., np.inf)
        
    elif ftype == "rayleigh":
        mean_true, var_true = rayleigh.fit(y1)
        mean_shifted, var_shifted = rayleigh.fit(y2)
        f_function = lambda x: np.sqrt(
            rayleigh.pdf(x, mean_true, var_true) *
            rayleigh.pdf(x, mean_shifted, var_shifted)
        )
        fidelity, _ = quad(f_function, -np.inf, np.inf)
        
    elif ftype == "poisson":
        k_max = max(max(y1) + 1, max(y2) + 1)
        k_values = np.arange(k_max)
        p_true = poisson.pmf(k_values, np.mean(y1))
        p_shifted = poisson.pmf(k_values, np.mean(y2))
        p_true /= np.sum(p_true)
        p_shifted /= np.sum(p_shifted)
        fidelity = np.sum(np.sqrt(p_true * p_shifted))
        
    else:
        valid_types = ["norm", "poisson", "gamma", "rayleigh"]
        raise ValueError(
            f"Invalid distribution type: '{ftype}'. "
            f"Supported types are: {', '.join(valid_types)}"
        )
    
    fisher_information = 8 * (1 - fidelity) / dx**2
    return fisher_information


def QFIM_pure(rho, drho):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum Fisher
    information matrix (QFIM) for pure states.

    For a pure state :math:`\rho = |\psi\rangle\langle\psi|`, the SLD takes
    the simple form :math:`L_a = 2\partial_a\rho`, allowing the QFIM entries
    to be computed as

    .. math::

        F_{ab} = \mathrm{Re}\,\mathrm{Tr}(\rho L_a L_b)
                = 4\,\mathrm{Re}\bigl(
                    \langle\partial_a\psi|\partial_b\psi\rangle
                    - \langle\partial_a\psi|\psi\rangle
                    \langle\psi|\partial_b\psi\rangle
                \bigr).

    Args:
        rho (np.array):
            Pure-state density matrix.
        drho (list):
            Derivatives of the density matrix with respect to the unknown
            parameters. Each element is a matrix of the same dimension as
            `rho`.

    Returns:
        (float/np.array):
            For single parameter estimation returns QFI (float).
            For multiparameter estimation returns QFIM (np.array).

    Raises:
        TypeError: If `drho` is not a list.
    """
    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    num_params = len(drho)
    if num_params == 1:
        sld = 2.0 * drho[0]
        return np.real(np.trace(rho @ sld @ sld))

    sld_list = [2.0 * d for d in drho]
    qfim = np.zeros((num_params, num_params))
    for i in range(num_params):
        for j in range(i, num_params):
            anticommutator = sld_list[i] @ sld_list[j] + sld_list[j] @ sld_list[i]
            qfim[i, j] = np.real(0.5 * np.trace(rho @ anticommutator))
            qfim[j, i] = qfim[i, j]
    return qfim


def QFIM_Kraus(rho0, K, dK, LDtype="SLD", exportLD=False, eps=None):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) for a quantum channel described by Kraus operators.

    The quantum channel is given by
    
    $$
    \rho=\sum_{i} K_i \rho_0 K_i^{\dagger},
    $$

    where $\rho_0$ is the initial state and $\{K_i\}$ are the Kraus operators.

    The derivatives of the density matrix $\partial_a\rho$ are calculated from the 
    derivatives of the Kraus operators $\{\partial_a K_i\}$ as
    
    $$
    \partial_a\rho=\sum_{i}\left[(\partial_a K_i)\rho_0 K_i^{\dagger}+K_i\rho_0(\partial_a K_i)^{\dagger}\right].
    $$

    Then the QFI (QFIM) is calculated via the function `QFIM` with the evolved state 
    $\rho$ and its derivatives $\{\partial_a\rho\}$.

    Args:
        rho0 (np.array): 
            Initial density matrix.
        K (list): 
            Kraus operators.
        dK (list): 
            Derivatives of the Kraus operators. It is a nested list where the first index 
            corresponds to the parameter and the second index corresponds to the Kraus operator index. 
            For example, `dK[0][1]` is the derivative of the second Kraus operator with respect 
            to the first parameter.
        LDtype (str, optional): 
            Types of QFI (QFIM) can be set as the objective function. Options:  
                - "SLD" (default): QFI (QFIM) based on symmetric logarithmic derivative.  
                - "RLD": QFI (QFIM) based on right logarithmic derivative.  
                - "LLD": QFI (QFIM) based on left logarithmic derivative.  
        exportLD (bool, optional): 
            Whether to export the values of logarithmic derivatives.  
        eps (float, optional): 
            Legacy relative spectral floor. When ``None`` the Julia 0.5
            relative precision policy is used.

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of dK is equal to one), the output is QFI 
            and for multiparameter estimation (the length of dK is more than one), it returns QFIM.
    """

    # Transpose dK: from [parameters][operators] to [operators][parameters]
    dK_transposed = [
        [dK[i][j] for i in range(len(K))] 
        for j in range(len(dK[0]))
    ]
    
    # Compute the evolved density matrix
    rho = sum(Ki @ rho0 @ Ki.conj().T for Ki in K)
    
    # Compute the derivatives of the density matrix
    drho = [
        sum(
            dKi @ rho0 @ Ki.conj().T + Ki @ rho0 @ dKi.conj().T
            for Ki, dKi in zip(K, dKj)
        )
        for dKj in dK_transposed
    ]
    
    return QFIM(rho, drho, LDtype=LDtype, exportLD=exportLD, eps=eps)


def QFIM_Bloch(r, dr, eps=1e-8):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) in Bloch representation.

    The Bloch vector representation of a quantum state is defined as
    
    $$
    \rho = \frac{1}{d}\left(\mathbb{I} + \sum_{i=1}^{d^2-1} r_i \lambda_i\right),
    $$
    
    where $\lambda_i$ are the generators of SU(d) group.

    Args:
        r (np.array): 
            Parameterized Bloch vector.
        dr (list): 
            Derivatives of the Bloch vector with respect to the unknown parameters. 
            Each element in the list is a vector of the same length as `r` and 
            represents the partial derivative of the Bloch vector with respect to 
            one parameter. For example, `dr[0]` is the derivative with respect to 
            the first parameter.
        eps (float, optional): 
            Machine epsilon.  

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of `dr` is equal to one), 
            the output is QFI and for multiparameter estimation (the length of `dr` 
            is more than one), it returns QFIM.

    Raises:
        TypeError: If `dr` is not a list.  
        ValueError: If the dimension of the Bloch vector is invalid.  
    """

    if not isinstance(dr, list):
        raise TypeError("dr must be a list of derivative vectors")

    num_params = len(dr)
    qfim_result = np.zeros((num_params, num_params))

    # Calculate dimension from Bloch vector length
    dim_float = np.sqrt(len(r) + 1)
    if dim_float.is_integer():
        dim = int(dim_float)
    else:
        raise ValueError("Invalid Bloch vector dimension")

    # Get SU(N) generators
    lambda_generators = suN_generator(dim)

    # Handle single-qubit system
    if dim == 2:
        r_norm = np.linalg.norm(r) ** 2
        
        # Pure state case
        if np.abs(r_norm - 1.0) < eps:
            for i in range(num_params):
                for j in range(i, num_params):
                    qfim_result[i, j] = np.real(np.inner(dr[i], dr[j]))
                    qfim_result[j, i] = qfim_result[i, j]
        # Mixed state case
        else:
            for i in range(num_params):
                for j in range(i, num_params):
                    term1 = np.inner(dr[i], dr[j])
                    term2 = (np.inner(r, dr[i]) * np.inner(r, dr[j])) / (1 - r_norm)
                    qfim_result[i, j] = np.real(term1 + term2)
                    qfim_result[j, i] = qfim_result[i, j]
    # Handle higher-dimensional systems
    else:
        # Reconstruct density matrix from Bloch vector
        rho = np.eye(dim, dtype=np.complex128) / dim
        for idx in range(dim**2 - 1):
            coeff = np.sqrt(dim * (dim - 1) / 2) * r[idx] / dim
            rho += coeff * lambda_generators[idx]

        # Calculate G matrix
        G = np.zeros((dim**2 - 1, dim**2 - 1), dtype=np.complex128)
        for i in range(dim**2 - 1):
            for j in range(i, dim**2 - 1):
                anticommutator = (
                    lambda_generators[i] @ lambda_generators[j] + 
                    lambda_generators[j] @ lambda_generators[i]
                )
                G[i, j] = 0.5 * np.trace(rho @ anticommutator)
                G[j, i] = G[i, j]

        # Calculate matrix for inversion
        r_vec = np.array(r).reshape(len(r), 1)
        mat = G * dim / (2 * (dim - 1)) - r_vec @ r_vec.T
        mat_inv = inv(mat)

        # Calculate QFIM
        for i in range(num_params):
            for j in range(i, num_params):
                dr_i = np.array(dr[i]).reshape(1, len(r))
                dr_j = np.array(dr[j]).reshape(len(r), 1)
                qfim_result[i, j] = np.real(dr_i @ mat_inv @ dr_j)[0, 0]
                qfim_result[j, i] = qfim_result[i, j]

    return qfim_result[0, 0] if num_params == 1 else qfim_result


def Williamson_form(sigma):
    r"""
    Perform the Williamson decomposition of a positive-definite
    :math:`2N\times 2N` covariance matrix :math:`\sigma`.

    Any positive-definite real symmetric matrix :math:`\sigma` can be
    decomposed as

    .. math::

        \sigma = S\,\mathrm{diag}(\nu_1,\dots,\nu_N,\nu_1,\dots,\nu_N)\,S^{\mathsf{T}},

    where :math:`S` is a symplectic matrix and :math:`\nu_k > 0` are the
    symplectic eigenvalues.

    Args:
        sigma (np.array):
            The :math:`2N\times 2N` covariance matrix (must be positive-definite).

    Returns:
        tuple:
            - S (np.array): The symplectic matrix.
            - c (np.array): The symplectic eigenvalues :math:`(\nu_1,\dots,\nu_N)`.
    """
    n = sigma.shape[0] // 2
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    sigma_sqrt = sqrtm(sigma)
    B = sigma_sqrt @ J @ sigma_sqrt
    T, Q = schur(B)
    eigenvalues = eigvals(T)
    c = np.array(sorted([ev.imag for ev in eigenvalues if ev.imag > 1e-10]))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(c))

    P = np.hstack([np.eye(2 * n)[:, 0::2], np.eye(2 * n)[:, 1::2]])
    temp1 = J @ sigma_sqrt @ Q
    temp2 = P @ np.block([[np.zeros((n, n)), -D_inv_sqrt], [D_inv_sqrt, np.zeros((n, n))]])
    S = np.linalg.inv((temp1 @ temp2).T) @ P.T
    return S, c


def QFIM_Gauss(R, dR, D, dD):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum 
    Fisher information matrix (QFIM) for Gaussian states.

    The Gaussian state is characterized by its first-order moment (displacement vector) 
    and second-order moment (covariance matrix). The QFIM is calculated using the 
    method described in [1].

    Args:
        R (np.array): 
            First-order moment (displacement vector).
        dR (list): 
            Derivatives of the first-order moment with respect to the unknown parameters. 
            Each element in the list is a vector of the same length as `R` and represents the partial 
            derivative of the displacement vector with respect to one parameter. For example, `dR[0]` 
            is the derivative with respect to the first parameter.
        D (np.array): 
            Second-order moment (covariance matrix).
        dD (list): 
            Derivatives of the second-order moment with respect to the unknown parameters. 
            Each element in the list is a matrix of the same dimension as `D` and 
            represents the partial derivative of the covariance matrix with respect to 
            one parameter. For example, `dD[0]` is the derivative with respect to 
            the first parameter.

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of `dR` is equal to one), 
            the output is QFI and for multiparameter estimation (the length of `dR` 
            is more than one), it returns QFIM.

    Notes:
        This function follows the approach from:
        [1] Monras, A., Phase space formalism for quantum estimation of Gaussian states, arXiv:1303.3682 (2013).
    """

    num_params = len(dR)
    m = len(R) // 2  # Number of modes
    qfim = np.zeros((num_params, num_params))

    # Compute the covariance matrix from the second-order moments and displacement
    cov_matrix = np.array(
        [
            [D[i][j] - R[i] * R[j] for j in range(2 * m)]
            for i in range(2 * m)
        ]
    )

    # Compute the derivatives of the covariance matrix
    dcov = []
    for k in range(num_params):
        dcov_k = np.zeros((2 * m, 2 * m))
        for i in range(2 * m):
            for j in range(2 * m):
                dcov_k[i, j] = dD[k][i][j] - dR[k][i] * R[j] - R[i] * dR[k][j]
        dcov.append(dcov_k)

    # Williamson decomposition of the covariance matrix
    S, c = Williamson_form(cov_matrix)

    # Define the basis matrices for the Gaussian representation
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.eye(2)
    a_gauss = [1j * sigma_y, sigma_z, identity, sigma_x]

    # Construct the basis matrices for the m-mode system
    es = []
    for i in range(m):
        row = []
        for j in range(m):
            e_ij = np.eye(1, m * m, m * i + j).reshape(m, m)
            row.append(e_ij)
        es.append(row)

    # As: a list of two-mode basis matrices for each of the four types
    As = []
    for a in a_gauss:
        A_type = []
        for i in range(m):
            for j in range(m):
                A_ij = np.kron(es[i][j], a) / np.sqrt(2)
                A_type.append(A_ij)
        As.append(A_type)

    # Compute the coefficients g for each parameter and each basis matrix
    g = []
    for k in range(num_params):
        g_k = []
        for A_type in As:
            g_type = []
            for A_mat in A_type:
                term = np.trace(inv(S) @ dcov[k] @ inv(S.T) @ A_mat.T)
                g_type.append(term)
            g_k.append(g_type)
        g.append(g_k)

    # Initialize the matrices G for each parameter
    G_matrices = [np.zeros((2 * m, 2 * m), dtype=np.complex128) for _ in range(num_params)]

    # Construct the matrices G for each parameter
    for k in range(num_params):
        for i in range(m):
            for j in range(m):
                for l in range(4):  # For each of the four types
                    denom = 4 * c[i] * c[j] + (-1) ** (l + 1)
                    A_l_ij = As[l][i * m + j]
                    G_matrices[k] += np.real(
                        g[k][l][i * m + j] / denom * inv(S.T) @ A_l_ij @ inv(S)
                    )

    # Compute the QFIM
    for i in range(num_params):
        for j in range(num_params):
            term1 = np.trace(G_matrices[i] @ dcov[j])
            term2 = dR[i] @ inv(cov_matrix) @ dR[j]
            qfim[i, j] = np.real(term1 + term2)

    if num_params == 1:
        return qfim[0, 0]
    else:
        return qfim
