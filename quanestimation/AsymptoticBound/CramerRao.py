import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.Common.Common import SIC, suN_generator
from scipy.integrate import quad
from scipy.stats import norm, poisson, rayleigh, gamma

def CFIM(rho, drho, M=[], eps=1e-8):
    r"""
    Calculation of the classical Fisher information matrix for the chosen measurements.

    This function computes the classical Fisher information (CFI) and classical Fisher 
    information matrix (CFIM) for a density matrix. The entry of CFIM $\mathcal{I}$
    is defined as

    $$
    \mathcal{I}_{ab}=\sum_y\frac{1}{p(y|\textbf{x})}[\partial_a p(y|\textbf{x})][\partial_b p(y|\textbf{x})],
    $$

    Symbols: 
        - $p(y|\textbf{x})=\mathrm{Tr}(\rho\Pi_y)$.
        - $\rho$: the parameterized density matrix.

    Args: 
        rho (np.array): 
            Density matrix.
        drho (list): 
            List of derivative matrices of the density matrix on the unknown 
            parameters to be estimated. For example, drho[0] is the derivative 
            matrix on the first parameter.
        M (list, optional): 
            List of positive operator-valued measure (POVM). The default 
            measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
        eps (float, optional): 
            Machine epsilon for numerical stability.

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of drho is equal to one), the output is CFI 
            and for multiparameter estimation (the length of drho is more than one), it returns CFIM.

    Raises:
        TypeError: If drho is not a list.
        TypeError: If M is not a list.   

    Example:
        rho = np.array([[0.5, 0], [0, 0.5]]);

        drho = [np.array([[1, 0], [0, -1]])];

        cfim = CFIM(rho, drho);     
    
    Notes: 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](https://www.physics.umb.edu/Research/QBism/solutions.html).
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
    Calculation of the classical Fisher information matrix (CFIM) for a given probability distributions.

    This function computes the classical Fisher information matrix (CFIM) for a given probability 
    distributions. The entry of FIM $I$ is defined as

    $$
    I_{ab}=\sum_{y}\frac{1}{p_y}[\partial_a p_y][\partial_b p_y],
    $$

    Symbols: 
        - $\{p_y\}$: a set of the discrete probability distribution.

    Args: 
        p (np.array): 
            The probability distribution.
        dp (list): 
            Derivatives of the probability distribution on the unknown parameters to 
            be estimated. For example, dp[0] is the derivative vector on the first parameter.
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of drho is equal to one), the output is CFI 
            and for multiparameter estimation (the length of drho is more than one), it returns CFIM.
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

    Args:
        data_true (np.array): 
            Experimental data obtained at the true parameter value.
        data_shifted (np.array): 
            Experimental data obtained at parameter value shifted by delta_x.
        delta_x (float): 
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

    This function computes the SLD operator $L_a$, which is determined by

    $$
    \partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)
    $$

    with $\rho$ the parameterized density matrix. The entries of SLD can be calculated as 

    $$
    \langle\lambda_i|L_{a}|\lambda_j\rangle=\frac{2\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle}{\lambda_i+\lambda_j
    $$

    for $\lambda_i~(\lambda_j) \neq 0$. If $\lambda_i=\lambda_j=0$, the entry of SLD is set to be zero.

    Args:
        rho (np.array): 
            Density matrix.
        drho (list): 
            Derivatives of the density matrix on the unknown parameters to be 
            estimated. For example, drho[0] is the derivative vector on the first parameter.
        rep (str, optional): 
            The basis for the SLDs. Options:  
                - "original" (default): basis same as input density matrix  
                - "eigen": basis same as eigenspace of density matrix
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (np.array/list): 
            For single parameter estimation (i.e., length of `drho` equals 1), returns a matrix.  
            For multiparameter estimation (i.e., length of `drho` is larger than 1), returns a list of matrices.

    Raises:
        TypeError: If `drho` is not a list.  
        ValueError: If `rep` has invalid value. 
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
    The RLD operator $\mathcal{R}_a$ is defined by

    $$
    \partial_{a}\rho=\rho \mathcal{R}_a
    $$

    with $\rho$ the parameterized density matrix. The entries of RLD can be calculated as 

    $$
    \langle\lambda_i| \mathcal{R}_{a} |\lambda_j\rangle=\frac{1}{\lambda_i}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    $$

    for $\lambda_i\neq 0$.

    Args:
        rho (np.array): 
            Density matrix.  
        drho (list):  
            Derivatives of the density matrix on the unknown parameters to be 
            estimated. For example, drho[0] is the derivative vector on the first parameter.
        rep (str, optional): 
            The basis for the RLD(s). Options:  
                - "original" (default): basis same as input density matrix.  
                - "eigen": basis same as eigenspace of density matrix.
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (np.array/list): 
            For single parameter estimation (i.e., length of `drho` equals 1), returns a matrix.  
            For multiparameter estimation (i.e., length of `drho` is larger than 1), returns a list of matrices.

    Raises:
        TypeError: If `drho` is not a list.
        ValueError: If `rep` has invalid value or RLD doesn't exist.
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

    The LLD operator $\mathcal{R}_a^{\dagger}$ is defined by

    $$
    \partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho.
    $$

    The entries of LLD can be calculated as 

    $$
    \langle\lambda_i| \mathcal{R}_{a}^{\dagger} |\lambda_j\rangle=\frac{1}{\lambda_j}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    $$

    for $\lambda_j\neq 0$.

    Args: 
        rho (np.array): 
            Density matrix.
        drho (list): 
            Derivatives of the density matrix on the unknown parameters to be estimated. 
            For example, drho[0] is the derivative vector on the first parameter.
        rep (str, optional): 
            The basis for the LLD(s). Options:  
                - "original" (default): basis same as input density matrix.  
                - "eigen": basis same as eigenspace of density matrix.
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (np.array/list): 
            For single parameter estimation (i.e., length of `drho` equals 1), returns a matrix.  
            For multiparameter estimation (i.e., length of `drho` is larger than 1), returns a list of matrices.

    Raises:
        TypeError: If `drho` is not a list.  
        ValueError: If `rep` has invalid value or LLD doesn't exist.  
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
    Calculate the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) for all types.

    The entry of QFIM $\mathcal{F}$ is defined as:

    $$
    \mathcal{F}_{ab}=\frac{1}{2}\mathrm{Tr}(\rho\{L_a, L_b\})
    $$

    with $L_a, L_b$ being SLD operators.

    Alternatively:

    $$
    \mathcal{F}_{ab}=\mathrm{Tr}(\rho \mathcal{R}_a \mathcal{R}^{\dagger}_b)
    $$

    with $\mathcal{R}_a$ being the RLD or LLD operator.

    Args:
        rho (np.array): 
            Density matrix.
        drho (list): 
            Derivatives of the density matrix with respect to the unknown parameters. 
            Each element in the list is a matrix of the same dimension as `rho` and 
            represents the partial derivative of the density matrix with respect to 
            one parameter. For example, `drho[0]` is the derivative with respect to 
            the first parameter.
        LDtype (str, optional): 
            Specifies the type of logarithmic derivative to use for QFI/QFIM calculation:  
                - "SLD": Symmetric Logarithmic Derivative (default).  
                - "RLD": Right Logarithmic Derivative.  
                - "LLD": Left Logarithmic Derivative.  
        exportLD (bool, optional): 
            Whether to export the values of logarithmic derivatives.  
        eps (float, optional): 
            Machine epsilon.  

    Returns:
        (float/np.array): 
            For single parameter estimation (i.e., length of `drho` equals 1), returns QFI.  
            For multiparameter estimation (i.e., length of `drho` is larger than 1), returns QFIM.  

    Raises:
        TypeError: If `drho` is not a list.
        ValueError: If `LDtype` is not one of the supported types ("SLD", "RLD", "LLD").        
    """

    if not isinstance(drho, list):
        raise TypeError("drho must be a list of derivative matrices")

    num_params = len(drho)
    qfim_result = None
    log_derivatives = None

    # Single parameter estimation
    if num_params == 1:
        if LDtype == "SLD":
            sld = SLD(rho, drho, eps=eps)
            anticommutator = sld @ sld + sld @ sld
            qfim_result = np.real(0.5 * np.trace(rho @ anticommutator))
        elif LDtype == "RLD":
            rld = RLD(rho, drho, eps=eps)
            qfim_result = np.real(np.trace(rho @ rld @ rld.conj().T))
        elif LDtype == "LLD":
            lld = LLD(rho, drho, eps=eps)
            qfim_result = np.real(np.trace(rho @ lld @ lld.conj().T))
        else:
            valid_types = ["SLD", "RLD", "LLD"]
            raise ValueError(
                f"Invalid LDtype: '{LDtype}'. Supported types: {', '.join(valid_types)}"
            )
        log_derivatives = sld if LDtype == "SLD" else rld if LDtype == "RLD" else lld

    # Multiparameter estimation
    else:
        if LDtype == "SLD":
            qfim_result = np.zeros((num_params, num_params))
            sld_list = SLD(rho, drho, eps=eps)
            for i in range(num_params):
                for j in range(i, num_params):
                    anticommutator = sld_list[i] @ sld_list[j] + sld_list[j] @ sld_list[i]
                    qfim_result[i, j] = np.real(0.5 * np.trace(rho @ anticommutator))
                    qfim_result[j, i] = qfim_result[i, j]
            log_derivatives = sld_list

        elif LDtype == "RLD":
            qfim_result = np.zeros((num_params, num_params), dtype=np.complex128)
            rld_list = RLD(rho, drho, eps=eps)
            for i in range(num_params):
                for j in range(i, num_params):
                    term = np.trace(rho @ rld_list[i] @ rld_list[j].conj().T)
                    qfim_result[i, j] = term
                    qfim_result[j, i] = term.conj()
            log_derivatives = rld_list

        elif LDtype == "LLD":
            qfim_result = np.zeros((num_params, num_params), dtype=np.complex128)
            lld_list = LLD(rho, drho, eps=eps)
            for i in range(num_params):
                for j in range(i, num_params):
                    term = np.trace(rho @ lld_list[i] @ lld_list[j].conj().T)
                    qfim_result[i, j] = term
                    qfim_result[j, i] = term.conj()
            log_derivatives = lld_list

        else:
            valid_types = ["SLD", "RLD", "LLD"]
            raise ValueError(
                f"Invalid LDtype: '{LDtype}'. Supported types: {', '.join(valid_types)}"
            )

    if exportLD:
        return qfim_result, log_derivatives
    return qfim_result


def QFIM_Kraus(rho0, K, dK, LDtype="SLD", exportLD=False, eps=1e-8):
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
            Machine epsilon.  

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

    # Compute the square root of the covariance matrix
    cov_sqrt = sqrtm(cov_matrix)

    # Define the symplectic matrix J for m modes
    J_block = np.array([[0, 1], [-1, 0]])
    J = np.kron(J_block, np.eye(m))

    # Compute the matrix B = cov_sqrt @ J @ cov_sqrt
    B = cov_sqrt @ J @ cov_sqrt

    # Permutation matrix to rearrange the basis
    P = np.eye(2 * m)
    # Rearrange the basis: first all q's then all p's
    P = np.vstack([P[::2], P[1::2]])

    # Schur decomposition of B
    _, Q = schur(B)
    eigenvalues = eigvals(B)
    # Extract the imaginary parts of every other eigenvalue
    c = eigenvalues[::2].imag

    # Diagonal matrix with entries 1/sqrt(c_i) for each mode
    diag_inv_sqrt = np.diagflat(1.0 / np.sqrt(c))

    # Construct the matrix S
    temp1 = J @ cov_sqrt @ Q
    temp2 = P @ np.kron(np.array([[0, 1], [-1, 0]]), -diag_inv_sqrt)
    S = inv(temp1 @ temp2).T @ P.T

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
