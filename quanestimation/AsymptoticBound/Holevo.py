
import numpy as np
from quanestimation.AsymptoticBound.CramerRao import QFIM
from scipy.linalg import sqrtm
# import cvxpy as cp

def Holevo_bound_trace(rho, drho, cost_function):
    """
    Description: Calculation the trace of Holevo Cramer Rao bound
                 for a density matrix.

    ---------
    Inputs
    ---------
    rho:
       --description: parameterized density matrix.
       --type: matrix

    drho:
       --description: derivatives of density matrix on all parameters to
                          be estimated. For example, drho[0] is the derivative
                          vector on the first parameter.
       --type: list (of matrix)

    C:
       --description: cost matrix in the Holevo bound.
       --type: matrix

    ----------
    Returns
    ----------
    Holevo trace:
        --description: trace of Holevo Cramer Rao bound. 
        --type: float number

    """

    if type(drho) != list:
        raise TypeError('Please make sure drho is list!')

    dim = len(rho)
    para_num = len(drho)
    CostG = cost_function

    QFIM_temp, SLD_temp = QFIM(rho, drho, rho_type='density_matrix', dtype='SLD', rep='original', exportLD=True)
    QFIMinv = np.linalg.inv(QFIM_temp)

    V = np.array([[0.+0.j for i in range(0,para_num)] for k in range(0,para_num)])
    for para_i in range(0,para_num):
        for para_j in range(0,para_num):
            Vij_temp = 0.+0.j
            for ki in range(0,para_num):
                for mi in range(0,para_num):
                    SLD_ki = SLD_temp[ki]
                    SLD_mi = SLD_temp[mi]
                    Vij_temp = Vij_temp+QFIMinv[para_i][ki]*QFIMinv[para_j][mi]\
                             *np.trace(np.dot(np.dot(rho, SLD_ki), SLD_mi))
            V[para_i][para_j] = Vij_temp

    real_part = np.dot(CostG,np.real(V))
    imag_part = np.dot(CostG,np.imag(V))
    Holevo_trace = np.trace(real_part)+np.trace(sqrtm(np.dot(imag_part,np.conj(np.transpose(imag_part)))))

    return Holevo_trace

# to be discussed and tested
# def Holevo_bound(rho, drho, C):
#     """
#     Description: Calculation the trace of Holevo Cramer Rao bound
#                  for a density matrix by semidefinite program (SDP).

#     ---------
#     Inputs
#     ---------
#     rho:
#        --description: parameterized density matrix.
#        --type: matrix

#     drho:
#        --description: derivatives of density matrix on all parameters to
#                           be estimated. For example, drho[0] is the derivative
#                           vector on the first parameter.
#        --type: list (of matrix)

#     C:
#        --description: cost matrix in the Holevo bound.
#        --type: matrix

#     ----------
#     Returns
#     ----------
#     Holevo bound:
#         --description: the Holevo bound solved by SDP. 
#         --type: matrix

#     """

#     if type(drho) != list:
#         raise TypeError('Please make sure drho is list!')
        
#     dim = len(rho)
#     para_num = len(drho)
    
#     #============optimization variables================
#     V = cp.Variable((para_num, para_num), symmetric=True)
    
#     X = []
#     for i in range(para_num):
#         X.append(cp.Variable((dim, dim), symmetric=True))
    
#     #================add constrains=================== 
#     constraints = [V[para_i][para_j] - cp.trace(rho@X[para_i]@X[para_j]) >= 0 for para_i in range(para_num) for para_j in range(para_num)]
#     constraints += [X[para_i]@drho[para_i]-np.eye(dim) == 0 for para_i in range(para_num) for para_i in range(para_num)]
#     constraints += [X[para_i]@drho[para_j] == 0 for para_i in range(para_num) for para_j in range(para_num) if para_i!=para_j]
    
#     prob = cp.Problem(cp.Minimize(cp.trace(C @ V)),constraints)
#     prob.solve()
    
#     return prob.value, X.value