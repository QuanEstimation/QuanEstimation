
import numpy as np
import scipy as sp
import cvxpy as cp

def Holevo_bound(rho, drho, C):
    """
    Description: Calculation the trace of Holevo Cramer Rao bound
                 for a density matrix by semidefinite program (SDP).

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
    Holevo bound:
        --description: the Holevo bound solved by semidefined programming. 
        --type: float

    """

    if type(drho) != list:
        raise TypeError('Please make sure drho is a list!')
        
    dim, r = len(rho), np.linalg.matrix_rank(rho)
    para_num = len(drho)
    
    Lambda = basis_lambda(rho)
    vec_drho = [[] for i in range(para_num)]
    for pi in range(para_num):
        vec_drho[pi] = np.array([np.real(np.trace(np.dot(drho[pi], Lambda[i]))) for i in range(len(Lambda))])

    num = 2*dim*r-r*r
    S = np.zeros((num,num),dtype=np.complex128)
    for a in range(num):
        for b in range(num):
            S[a][b] = np.trace(np.dot(Lambda[a], np.dot(Lambda[b], rho)))

    # how to decompose S when S is semidefinite matrix ???
    # R = np.linalg.cholesky(S).conj().T
    lu, d, perm = sp.linalg.ldl(S.round(8))
    R = np.dot(lu, sp.linalg.sqrtm(d)).conj().T
    #============optimization variables================
    V = cp.Variable((para_num, para_num), symmetric=True)
    X = cp.Variable((num, para_num))
    #================add constrains=================== 
    constraints = [cp.bmat([[V, X.T @ R.conj().T], [R @ X, np.identity(num)]]) >> 0]
    for i in range(para_num):
        for j in range(para_num):
            if i==j:
                constraints += [X[:,i].T @ vec_drho[j] == 1]
            else:
                constraints += [X[:,i].T @ vec_drho[j] == 0]
    
    prob = cp.Problem(cp.Minimize(cp.trace(C @ V)),constraints)
    prob.solve()
    
    return prob.value, X.value

def basis_lambda(rho):
    val, vec = np.linalg.eig(rho)
    dim, r = len(rho), np.linalg.matrix_rank(rho)
    basis = [vec[:,i].reshape(dim,1) for i in range(len(vec))]
    Lambda = [[] for i in range(2*dim*r-r*r)]
    for i in range(r):
        Lambda[i] = np.dot(basis[i],basis[i].conj().T)
    for j in range(1,r):
        for i in range(j):
            ind1 = int(r+0.5*(j*j-j)+i)
            ind2 = int(0.5*(r*r+r)+0.5*(j*j-j)+i)
            Lambda[ind1] = (np.dot(basis[i],basis[j].conj().T)+np.dot(basis[j],basis[i].conj().T))/np.sqrt(2.0)
            Lambda[ind2] = 1.0j*(np.dot(basis[i],basis[j].conj().T)-np.dot(basis[j],basis[i].conj().T))/np.sqrt(2.0)    
    
    for i in range(r):
        for k in range(r,dim):
            ind3 = int(r*k+i)
            ind4 = int(r*dim-r*r+r*k+i)
            Lambda[ind3] = (np.dot(basis[i],basis[k].conj().T)+np.dot(basis[k],basis[i].conj().T))/np.sqrt(2.0)
            Lambda[ind4] = 1.0j*(np.dot(basis[i],basis[k].conj().T)-np.dot(basis[k],basis[i].conj().T))/np.sqrt(2.0)    
    
    return Lambda
