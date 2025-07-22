import numpy as np


def Kraus(rho0, K, dK):
    r"""
    Parameterization of a quantum state using Kraus operators.

    The evolved density matrix $\rho$ is given by
    
    \begin{aligned}
        \rho=\sum_i K_i \rho_0 K_i^{\dagger},
    \end{aligned}

    where $\rho_0$ is the initial density matrix and $K_i$ are the Kraus operators.

    Args: 
        rho0 (np.array): 
            Initial density matrix.
        K (list): 
            Kraus operators.
        dK (list): 
            Derivatives of the Kraus operators with respect to the unknown parameters to be 
            estimated. This is a nested list where the first index corresponds to the Kraus operator 
            and the second index corresponds to the parameter. For example, `dK[0][1]` is the derivative 
            of the second Kraus operator with respect to the first parameter.

    Returns:
        (tuple):
            rho (np.array): 
                Evolved density matrix.
                
            drho (list): 
                Derivatives of the evolved density matrix with respect to the unknown parameters.  
                Each element in the list is a matrix representing the partial derivative of $\rho$ with 
                respect to one parameter.
    """

    k_num = len(K)
    para_num = len(dK[0])
    dK_reshape = [[dK[i][j] for i in range(k_num)] for j in range(para_num)]

    rho = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K])
    drho = [sum([(np.dot(dKi, np.dot(rho0, Ki.conj().T))+ np.dot(Ki, np.dot(rho0, dKi.conj().T))) for (Ki, dKi) in zip(K, dKj)]) for dKj in dK_reshape]

    return rho, drho
