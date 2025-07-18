import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.Common.Common import SIC, suN_generator
from scipy.integrate import quad
from scipy.stats import norm, poisson, rayleigh, gamma

def CFIM(rho, drho, M=[], eps=1e-8):
    r"""
    Calculation of the classical Fisher information (CFI) and classical Fisher 
    information matrix (CFIM) for a density matrix. The entry of CFIM $\mathcal{I}$
    is defined as
    \begin{align}
    \mathcal{I}_{ab}=\sum_y\frac{1}{p(y|\textbf{x})}[\partial_a p(y|\textbf{x})][\partial_b p(y|\textbf{x})],
    \end{align}

    where $p(y|\textbf{x})=\mathrm{Tr}(\rho\Pi_y)$ with $\rho$ the parameterized 
    density matrix.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM). 

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **CFI (CFIM):** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is CFI and for multiparameter estimation (the length of drho 
        is more than one), it returns CFIM.
    
    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    if not isinstance(drho, list):
        raise TypeError("Please make sure drho is a list!")

    if not M:
        M = SIC(len(rho[0]))
    else:
        if not isinstance(M, list):
            raise TypeError("Please make sure M is a list!")

    num_measurements = len(M)
    num_params = len(drho)
    cfim_res = np.zeros([num_params, num_params])
    
    for i in range(num_measurements):
        povm_element = M[i]
        p = np.real(np.trace(rho @ povm_element))
        c_add = np.zeros([num_params, num_params])
        
        if p > eps:
            for param_i in range(num_params):
                drho_i = drho[param_i]
                dp_i = np.real(np.trace(drho_i @ povm_element))
                
                for param_j in range(param_i, num_params):
                    drho_j = drho[param_j]
                    dp_j = np.real(np.trace(drho_j @ povm_element))
                    c_add[param_i][param_j] = np.real(dp_i * dp_j / p)
                    c_add[param_j][param_i] = np.real(dp_i * dp_j / p)
                    
        cfim_res += c_add

    if num_params == 1:
        return cfim_res[0][0]
    else:
        return cfim_res


def FIM(p, dp, eps=1e-8):
    r"""
    Calculation of the classical Fisher information (CFI) and classical Fisher 
    information matrix (CFIM) for classical scenarios. The entry of FIM $I$
    is defined as
    \begin{align}
    I_{ab}=\sum_{y}\frac{1}{p_y}[\partial_a p_y][\partial_b p_y],
    \end{align}

    where $\{p_y\}$ is a set of the discrete probability distribution.

    Parameters
    ----------
    > **p:** `array` 
        -- The probability distribution.

    > **dp:** `list`
        -- Derivatives of the probability distribution on the unknown parameters to 
        be estimated. For example, dp[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **CFI (CFIM):** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is CFI and for multiparameter estimation (the length of drho 
        is more than one), it returns CFIM.
    """

    num_params = len(dp)
    num_measurements = len(p)
    fim_matrix = np.zeros([num_params, num_params])
    
    for outcome_idx in range(num_measurements):
        p_value = p[outcome_idx]
        fim_add = np.zeros([num_params, num_params])
        
        if p_value > eps:
            for param_i in range(num_params):
                dp_i = dp[param_i][outcome_idx]
                
                for param_j in range(param_i, num_params):
                    dp_j = dp[param_j][outcome_idx]
                    term = np.real(dp_i * dp_j / p_value)
                    fim_add[param_i][param_j] = term
                    fim_add[param_j][param_i] = term
                    
        fim_matrix += fim_add

    if num_params == 1:
        return fim_matrix[0][0]
    else:
        return fim_matrix

def FI_Expt(data_true, data_shifted, delta_x, ftype="norm"):
    """
    Calculate the classical Fisher information (CFI) based on experimental data.

    Parameters
    ----------
    data_true : array
        Experimental data obtained at the true parameter value.
    data_shifted : array
        Experimental data obtained at parameter value shifted by delta_x.
    delta_x : float
        Small known parameter shift.
    distribution_type : str, optional
        Probability distribution of the data. Options:
        - "norm": normal distribution (default)
        - "gamma": gamma distribution
        - "rayleigh": Rayleigh distribution
        - "poisson": Poisson distribution

    Returns
    -------
    float
        Classical Fisher information

    Note
    ----
    The current implementation may be unstable and is subject to future modification.
    """
    fidelity = 0.0
    if ftype == "norm":
        mu_true, std_true = norm.fit(data_true)
        mu_shifted, std_shifted = norm.fit(data_shifted)
        f_function = lambda x: np.sqrt(
            norm.pdf(x, mu_true, std_true) * norm.pdf(x, mu_shifted, std_shifted)
        )
        fidelity, _ = quad(f_function, -np.inf, np.inf)
        
    elif ftype == "gamma":
        a_true, alpha_true, beta_true = gamma.fit(data_true)
        a_shifted, alpha_shifted, beta_shifted = gamma.fit(data_shifted)
        f_function = lambda x: np.sqrt(
            gamma.pdf(x, a_true, alpha_true, beta_true) *
            gamma.pdf(x, a_shifted, alpha_shifted, beta_shifted)
        )
        fidelity, _ = quad(f_function, 0., np.inf)
        
    elif ftype == "rayleigh":
        mean_true, var_true = rayleigh.fit(data_true)
        mean_shifted, var_shifted = rayleigh.fit(data_shifted)
        f_function = lambda x: np.sqrt(
            rayleigh.pdf(x, mean_true, var_true) *
            rayleigh.pdf(x, mean_shifted, var_shifted)
        )
        fidelity, _ = quad(f_function, -np.inf, np.inf)
        
    elif ftype == "poisson":
        k_max = max(max(data_true) + 1, max(data_shifted) + 1)
        k_values = np.arange(k_max)
        p_true = poisson.pmf(k_values, np.mean(data_true))
        p_shifted = poisson.pmf(k_values, np.mean(data_shifted))
        p_true /= np.sum(p_true)
        p_shifted /= np.sum(p_shifted)
        fidelity = np.sum(np.sqrt(p_true * p_shifted))
        
    else:
        valid_types = ["norm", "poisson", "gamma", "rayleigh"]
        raise ValueError(
            f"Invalid distribution type: '{ftype}'. "
            f"Supported types are: {', '.join(valid_types)}"
        )
    
    fisher_information = 8 * (1 - fidelity) / delta_x**2
    return fisher_information


def SLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the symmetric logarithmic derivative (SLD) for a density matrix.
    The SLD operator $L_a$ is determined by
    \begin{align}
    \partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)
    \end{align}

    with $\rho$ the parameterized density matrix. The entries of SLD can be calculated
    as 
    \begin{align}
    \langle\lambda_i|L_{a}|\lambda_j\rangle=\frac{2\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle}{\lambda_i+\lambda_j}
    \end{align}

    for $\lambda_i~(\lambda_j) \neq 0$. If $\lambda_i=\lambda_j=0$, the entry of SLD is set to be zero.

    Parameters
    ----------
    rho : matrix
        Density matrix.

    drho : list
        Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    rep : string, optional
        The basis for the SLDs. Options are:  
        "original" (default) - basis same as input density matrix  
        "eigen" - basis same as eigenspace of density matrix

    eps : float, optional
        Machine epsilon (default: 1e-8)

    Returns
    -------
    matrix or list
        For single parameter estimation (len(drho)=1), returns a matrix.
        For multiparameter estimation (len(drho)>1), returns a list of matrices.

    Raises
    ------
    TypeError
        If drho is not a list
    ValueError
        If rep has invalid value
    """

    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    num_params = len(drho)
    dim = len(rho)
    slds = [None] * num_params

    purity = np.trace(rho @ rho)

    # Handle pure state case
    if np.abs(1 - purity) < eps:
        sld_original = [2 * d for d in drho]
        
        for i in range(num_params):
            if rep == "original":
                slds[i] = sld_original[i]
            elif rep == "eigen":
                eigenvalues, eigenvectors = np.linalg.eig(rho)
                eigenvalues = np.real(eigenvalues)
                slds[i] = eigenvectors.conj().T @ sld_original[i] @ eigenvectors
            else:
                valid_reps = ["original", "eigen"]
                raise ValueError(f"Invalid rep value: '{rep}'. Valid options: {valid_reps}")
        
        return slds[0] if num_params == 1 else slds

    # Handle mixed state case
    eigenvalues, eigenvectors = np.linalg.eig(rho)
    eigenvalues = np.real(eigenvalues)
    
    for param_idx in range(num_params):
        sld_eigenbasis = np.zeros((dim, dim), dtype=np.complex128)
        
        for i in range(dim):
            for j in range(dim):
                if eigenvalues[i] + eigenvalues[j] > eps:
                    # Calculate matrix element in eigenbasis
                    numerator = 2 * (eigenvectors[:, i].conj().T @ drho[param_idx] @ eigenvectors[:, j])
                    sld_eigenbasis[i, j] = numerator / (eigenvalues[i] + eigenvalues[j])
        
        # Handle any potential infinities
        sld_eigenbasis[np.isinf(sld_eigenbasis)] = 0.0
        
        # Transform to requested basis
        if rep == "original":
            slds[param_idx] = eigenvectors @ sld_eigenbasis @ eigenvectors.conj().T
        elif rep == "eigen":
            slds[param_idx] = sld_eigenbasis
        else:
            valid_reps = ["original", "eigen"]
            raise ValueError(f"Invalid rep value: '{rep}'. Valid options: {valid_reps}")
    
    return slds[0] if num_params == 1 else slds


def RLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the right logarithmic derivative (RLD) for a density matrix.
    The RLD operator defined by $\partial_{a}\rho=\rho \mathcal{R}_a$
    with $\rho$ the parameterized density matrix. 
    \begin{align}
    \langle\lambda_i| \mathcal{R}_{a} |\lambda_j\rangle=\frac{1}{\lambda_i}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    \end{align}

    for $\lambda_i\neq 0$ is the $ij$th entry of RLD.

    Parameters
    ----------
    rho : matrix
        Density matrix.
    drho : list
        Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.
    rep : string, optional
        The basis for the RLD(s). Options are:  
        "original" (default) - basis same as input density matrix  
        "eigen" - basis same as eigenspace of density matrix
    eps : float, optional
        Machine epsilon (default: 1e-8)

    Returns
    -------
    matrix or list
        For single parameter estimation (len(drho)=1), returns a matrix.
        For multiparameter estimation (len(drho)>1), returns a list of matrices.

    Raises
    ------
    TypeError
        If drho is not a list
    ValueError
        If rep has invalid value or RLD doesn't exist
    """
    
    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    num_params = len(drho)
    dim = len(rho)
    rld_list = [None] * num_params

    eigenvalues, eigenvectors = np.linalg.eig(rho)
    eigenvalues = np.real(eigenvalues)
    
    for param_idx in range(num_params):
        rld_eigenbasis = np.zeros((dim, dim), dtype=np.complex128)
        
        for i in range(dim):
            for j in range(dim):
                # Calculate matrix element in eigenbasis
                element = (
                    eigenvectors[:, i].conj().T 
                    @ drho[param_idx] 
                    @ eigenvectors[:, j]
                )
                
                if np.abs(eigenvalues[i]) > eps:
                    rld_eigenbasis[i, j] = element / eigenvalues[i]
                else:
                    if np.abs(element) > eps:
                        raise ValueError(
                            "RLD does not exist. It only exists when the support of "
                            "drho is contained in the support of rho."
                        )
        
        # Handle any potential infinities
        rld_eigenbasis[np.isinf(rld_eigenbasis)] = 0.0
        
        # Transform to requested basis
        if rep == "original":
            rld_list[param_idx] = (
                eigenvectors 
                @ rld_eigenbasis 
                @ eigenvectors.conj().T
            )
        elif rep == "eigen":
            rld_list[param_idx] = rld_eigenbasis
        else:
            valid_reps = ["original", "eigen"]
            raise ValueError(
                f"Invalid rep value: '{rep}'. Valid options: {', '.join(valid_reps)}"
            )
    
    return rld_list[0] if num_params == 1 else rld_list


def LLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the left logarithmic derivative (LLD) for a density matrix $\rho$.
    The LLD operator is defined by $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$. 
    The entries of LLD can be calculated as 
    \begin{align}
    \langle\lambda_i| \mathcal{R}_{a}^{\dagger} |\lambda_j\rangle=\frac{1}{\lambda_j}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    \end{align}

    for $\lambda_j\neq 0$.

    Parameters
    ----------
    rho : matrix
        Density matrix.
    drho : list
        Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.
    rep : string, optional
        The basis for the LLD(s). Options are:  
        "original" (default) - basis same as input density matrix  
        "eigen" - basis same as eigenspace of density matrix
    eps : float, optional
        Machine epsilon (default: 1e-8)

    Returns
    -------
    matrix or list
        For single parameter estimation (len(drho)=1), returns a matrix.
        For multiparameter estimation (len(drho)>1), returns a list of matrices.

    Raises
    ------
    TypeError
        If drho is not a list
    ValueError
        If rep has invalid value or LLD doesn't exist
    """

    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    param_num = len(drho)
    dim = len(rho)
    lld_list = [None] * param_num

    eigenvalues, eigenvectors = np.linalg.eig(rho)
    eigenvalues = np.real(eigenvalues)
    
    for param_idx in range(param_num):
        lld_eigenbasis = np.zeros((dim, dim), dtype=np.complex128)
        
        for i in range(dim):
            for j in range(dim):
                # Calculate matrix element in eigenbasis
                element = (
                    eigenvectors[:, i].conj().T 
                    @ drho[param_idx] 
                    @ eigenvectors[:, j]
                )
                
                if np.abs(eigenvalues[j]) > eps:
                    lld_eigenbasis[i, j] = element / eigenvalues[j]
                else:
                    if np.abs(element) > eps:
                        raise ValueError(
                            "LLD does not exist. It only exists when the support of "
                            "drho is contained in the support of rho."
                        )
        
        # Handle any potential infinities
        lld_eigenbasis[np.isinf(lld_eigenbasis)] = 0.0
        
        # Transform to requested basis
        if rep == "original":
            lld_list[param_idx] = (
                eigenvectors 
                @ lld_eigenbasis 
                @ eigenvectors.conj().T
            )
        elif rep == "eigen":
            lld_list[param_idx] = lld_eigenbasis
        else:
            valid_reps = ["original", "eigen"]
            raise ValueError(
                f"Invalid rep value: '{rep}'. Valid options: {', '.join(valid_reps)}"
            )
    
    return lld_list[0] if param_num == 1 else lld_list


def QFIM(rho, drho, LDtype="SLD", exportLD=False, eps=1e-8):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) for all types. The entry of QFIM $\mathcal{F}$
    is defined as
    \begin{align}
    \mathcal{F}_{ab}=\frac{1}{2}\mathrm{Tr}(\rho\{L_a, L_b\})
    \end{align}

    with $L_a, L_b$ are SLD operators and 

    and 
    \begin{align}
    \mathcal{F}_{ab}=\mathrm{Tr}(\rho \mathcal{R}_a \mathcal{R}^{\dagger}_b)
    \end{align}

    with $\mathcal{R}_a$ the RLD or LLD operator.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **exportLD:** `bool`
        -- Whether or not to export the values of logarithmic derivatives. If set True
        then the the values of logarithmic derivatives will be exported.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM:** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list")

    para_num = len(drho)

    # single parameter estimation
    if para_num == 1:
        if LDtype == "SLD":
            LD_tp = SLD(rho, drho, eps=eps)
            SLD_ac = LD_tp @ LD_tp + LD_tp @ LD_tp
            QFIM_res = np.real(0.5 * np.trace(rho @ SLD_ac))
        elif LDtype == "RLD":
            LD_tp = RLD(rho, drho, eps=eps)
            QFIM_res = np.real(np.trace(rho @ LD_tp @ LD_tp.conj().transpose()))
        elif LDtype == "LLD":
            LD_tp = LLD(rho, drho, eps=eps)
            QFIM_res = np.real(np.trace(rho @ LD_tp @ LD_tp.conj().transpose()))
        else:
            raise ValueError("{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(LDtype))

    # multiparameter estimation
    else:
        if LDtype == "SLD":
            QFIM_res = np.zeros([para_num, para_num])
            LD_tp = SLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    SLD_ac = LD_tp[para_i] @ LD_tp[para_j] + LD_tp[para_j] @ LD_tp[para_i]
                    QFIM_res[para_i][para_j] = np.real(0.5 * np.trace(rho @ SLD_ac))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        elif LDtype == "RLD":
            QFIM_res = np.zeros((para_num, para_num), dtype=np.complex128)
            LD_tp = RLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.trace(rho @ LD_tp[para_i] @ LD_tp[para_j].conj().transpose())
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j].conj()
        elif LDtype == "LLD":
            QFIM_res = np.zeros((para_num, para_num), dtype=np.complex128)
            LD_tp = LLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.trace(rho @ LD_tp[para_i] @ LD_tp[para_j].conj().transpose())
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j].conj()
        else:
            raise ValueError("{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(LDtype))

    if exportLD == False:
        return QFIM_res
    else:
        return QFIM_res, LD_tp


def QFIM_Kraus(rho0, K, dK, LDtype="SLD", exportLD=False, eps=1e-8):
    """
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) with Kraus operator(s) for all types.

    Parameters
    ----------
    > **rho0:** `matrix`
        -- Initial state (density matrix).

    > **K:** `list`
        -- Kraus operator(s).

    > **dK:** `list` 
        -- Derivatives of the Kraus operator(s) on the unknown parameters to be 
        estimated.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **exportLD:** `bool`
        -- Whether or not to export the values of logarithmic derivatives. If set True
        then the the values of logarithmic derivatives will be exported.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    dK = [[dK[i][j] for i in range(len(K))] for j in range(len(dK[0]))]
    rho = sum([Ki @ rho0 @ Ki.conj().T for Ki in K])
    drho = [
                sum(
                    [
                        (dKi @ rho0 @ Ki.conj().T+ Ki @ rho0 @ dKi.conj().T)
                        for (Ki, dKi) in zip(K, dKj)
                    ]
                )
                for dKj in dK
            ]
    return QFIM(rho, drho, LDtype=LDtype, exportLD=exportLD, eps=eps)


def QFIM_Bloch(r, dr, eps=1e-8):
    """
    Calculation of the SLD based quantum Fisher information (QFI) and quantum  
    Fisher information matrix (QFIM) in Bloch representation.

    Parameters
    ----------
    > **r:** `np.array`
        -- Parameterized Bloch vector.

    > **dr:** `list `
        -- Derivatives of the Bloch vector on the unknown parameters to be 
        estimated. For example, dr[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM in Bloch representation:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    if type(dr) != list:
        raise TypeError("Please make sure dr is a list")

    para_num = len(dr)
    QFIM_res = np.zeros([para_num, para_num])

    dim_float = np.sqrt(len(r) + 1)
    if dim_float.is_integer():
        dim = int(dim_float)
    else:
        raise ValueError("The dimension of the Bloch vector is wrong")

    Lambda = suN_generator(dim)

    if dim == 2:
        #### single-qubit system ####
        r_norm = np.linalg.norm(r) ** 2
        if np.abs(r_norm - 1.0) < eps:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(np.inner(dr[para_i], dr[para_j]))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        else:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(
                        np.inner(dr[para_i], dr[para_j])
                        + np.inner(r, dr[para_i])
                        * np.inner(r, dr[para_j])
                        / (1 - r_norm)
                    )
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
    else:
        rho = np.identity(dim, dtype=np.complex128) / dim
        for di in range(dim**2 - 1):
            rho += np.sqrt(dim * (dim - 1) / 2) * r[di] * Lambda[di] / dim

        G = np.zeros((dim**2 - 1, dim**2 - 1), dtype=np.complex128)
        for row_i in range(dim**2 - 1):
            for col_j in range(row_i, dim**2 - 1):
                anti_commu = Lambda[row_i] @ Lambda[col_j] + Lambda[col_j] @ Lambda[row_i]
                G[row_i][col_j] = 0.5 * np.trace(rho @ anti_commu)
                G[col_j][row_i] = G[row_i][col_j]

        mat_tp = G * dim / (2 * (dim - 1)) - np.dot(
            np.array(r).reshape(len(r), 1), np.array(r).reshape(1, len(r))
        )
        mat_inv = inv(mat_tp)

        for para_m in range(0, para_num):
            for para_n in range(para_m, para_num):
                QFIM_res[para_m][para_n] = np.real(
                    np.dot(
                        np.array(dr[para_n]).reshape(1, len(r)),
                        np.dot(mat_inv, np.array(dr[para_m]).reshape(len(r), 1)),
                    )[0][0]
                )
                QFIM_res[para_n][para_m] = QFIM_res[para_m][para_n]

    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res


def QFIM_Gauss(R, dR, D, dD):
    """
    Calculation of the SLD based quantum Fisher information (QFI) and quantum 
    Fisher information matrix (QFIM) with gaussian states.

    Parameters
    ----------
    > **R:** `array` 
        -- First-order moment.

    > **dR:** `list`
        -- Derivatives of the first-order moment on the unknown parameters to be 
        estimated. For example, dR[0] is the derivative vector on the first 
        parameter.

    > **D:** `matrix`
        -- Second-order moment.

    > **dD:** `list`
        -- Derivatives of the second-order moment on the unknown parameters to be 
        estimated. For example, dD[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM with gaussian states:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    para_num = len(dR)
    m = int(len(R) / 2)
    QFIM_res = np.zeros([para_num, para_num])

    C = np.array(
        [
            [D[i][j] - R[i] * R[j] for j in range(2 * m)]
            for i in range(2 * m)
        ]
    )
    dC = [
        np.array(
            [
                [
                    dD[k][i][j] - dR[k][i] * R[j] - R[i] * dR[k][j]
                    for j in range(2 * m)
                ]
                for i in range(2 * m)
            ]
        )
        for k in range(para_num)
    ]

    C_sqrt = sqrtm(C)
    J = np.kron([[0, 1], [-1, 0]], np.eye(m))
    B = C_sqrt @ J @ C_sqrt
    P = np.eye(2 * m)
    P = np.vstack([P[:][::2], P[:][1::2]])
    T, Q = schur(B)
    vals = eigvals(B)
    c = vals[::2].imag
    Diag = np.diagflat(c**-0.5)
    S = inv(J @ C_sqrt @ Q @ P @ np.kron([[0, 1], [-1, 0]], -Diag)).T @ P.T

    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    a_Gauss = [1j * sy, sz, np.eye(2), sx]

    es = [
        [np.eye(1, m**2, m * i + j).reshape(m, m) for j in range(m)] for i in range(m)
    ]

    As = [[np.kron(s, a_Gauss[i]) / np.sqrt(2) for s in es] for i in range(4)]
    gs = [
        [[[np.trace(inv(S) @ dC @ inv(S.T) @ aa.T) for aa in a] for a in A] for A in As]
        for dC in dC
    ]
    G = [np.zeros((2 * m, 2 * m)).astype(np.longdouble) for _ in range(para_num)]

    for i in range(para_num):
        for j in range(m):
            for k in range(m):
                for l in range(4):
                    G[i] += np.real(
                        gs[i][l][j][k]
                        / (4 * c[j] * c[k] + (-1) ** (l + 1))
                        * inv(S.T)
                        @ As[l][j][k]
                        @ inv(S)
                    )

    QFIM_res += np.real(
        [
            [np.trace(G[i] @ dC[j]) + dR[i] @ inv(C) @ dR[j] for j in range(para_num)]
            for i in range(para_num)
        ]
    )

    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res
