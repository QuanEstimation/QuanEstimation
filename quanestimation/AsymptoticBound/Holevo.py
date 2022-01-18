
import numpy as np
import scipy as sp
import cvxpy as cp
from quanestimation.Common import suN_generator
from quanestimation.AsymptoticBound.CramerRao import QFIM

def Holevo_bound(rho, drho, W, accuracy=1e-8):
    """
    Description: solve Holevo Cramer-Rao bound via the semidefinite program (SDP).
                 
    ---------
    Inputs
    ---------
    rho:
        --description: parameterized density matrix.
        --type: matrix

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, drho[0] is the derivative 
                       vector on the first parameter.
        --type: list (of matrix)

    W:
       --description: cost matrix.
       --type: matrix

    ----------
    Returns
    ----------
    Holevo bound:
        --description: the Holevo Cramer-Rao bound. 
        --type: float

    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    if len(drho) == 1:
        print("In single parameter scenario, HCRB is equivalent to QFI. This function will return the value of QFI")
        f = QFIM(rho, drho, accuracy=accuracy)
        return f
    else:
        dim = len(rho)
        num = dim*dim
        para_num = len(drho)
    
        Lambda = [np.identity(dim)] + suN_generator(dim)
        Lambda = Lambda/np.sqrt(2)

        vec_drho = [[] for i in range(para_num)]
        for pi in range(para_num):
            vec_drho[pi] = np.array([np.real(np.trace(np.dot(drho[pi],Lambda[i]))) for i in range(len(Lambda))])

        S = np.zeros((num,num),dtype=np.complex128)
        for a in range(num):
            for b in range(num):
                S[a][b] = np.trace(np.dot(Lambda[a], np.dot(Lambda[b], rho)))

        accu = len(str(int(1/accuracy)))-1
        lu, d, perm = sp.linalg.ldl(S.round(accu))
        R = np.dot(lu, sp.linalg.sqrtm(d)).conj().T
        #============optimization variables================
        V = cp.Variable((para_num, para_num))
        X = cp.Variable((num, para_num))
        #================add constraints=================== 
        constraints = [cp.bmat([[V, X.T @ R.conj().T], [R @ X, np.identity(num)]]) >> 0]
        for i in range(para_num):
            for j in range(para_num):
                if i==j:
                    constraints += [X[:,i].T @ vec_drho[j] == 1]
                else:
                    constraints += [X[:,i].T @ vec_drho[j] == 0]
    
        prob = cp.Problem(cp.Minimize(cp.trace(W @ V)), constraints)
        prob.solve()
    
        return prob.value, X.value, V.value
