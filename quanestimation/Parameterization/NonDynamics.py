import numpy as np


def Kraus(rho0, K, dK):
    r"""
    The parameterization of a state is
    \begin{align}
    \rho=\sum_i K_i\rho_0K_i^{\dagger},
    \end{align} 

    where $\rho$ is the evolved density matrix, $K_i$ is the Kraus operator.

    Parameters
    ----------
    > **K:** `list`
        -- Kraus operators.

    > **dK:** `list`
        -- Derivatives of the Kraus operators with respect to the unknown parameters to be 
        estimated. For example, dK[0] is the derivative vector on the first 
        parameter.

    > **rho0:** `matrix`
        -- Initial state (density matrix).

    Returns
    ----------
    Density matrix and its derivatives on the unknown parameters.
    """

    k_num = len(K)
    para_num = len(dK[0])
    dK_reshape = [[dK[i][j] for i in range(k_num)] for j in range(para_num)]

    rho = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K])
    drho = [sum([(np.dot(dKi, np.dot(rho0, Ki.conj().T))+ np.dot(Ki, np.dot(rho0, dKi.conj().T))) for (Ki, dKi) in zip(K, dKj)]) for dKj in dK_reshape]

    return rho, drho
    
    