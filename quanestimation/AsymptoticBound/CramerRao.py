import numpy as np
import numpy.linalg as LA
from numpy.linalg import inv
from scipy.integrate import simps
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.Common.common import SIC, extract_ele
#===============================================================================
#Subclass: metrology
#===============================================================================
"""
calculation of classical Fisher information matrix and quantum
Fisher information matrix.
"""
       
def CFIM(rho, drho, M=[], eps=1e-8):
    """
    Description: Calculation classical Fisher information matrix (CFIM)
                 for a density matrix.

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

    M:
       --description: a set of POVM. It takes the form [M1, M2, ...].
       --type: list (of matrix)

    ----------
    Returns
    ----------
    CFIM:
        --description: classical Fisher information matrix. If the length
                       of drho is one, the output is a float number (CFI),
                       otherwise it returns a matrix (CFIM).
        --type: float number (CFI) or matrix (CFIM)

    """
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    if M==[]: 
        M = SIC(len(rho[0]))
    else:
        if type(M) != list:
            raise TypeError("Please make sure M is a list!")

    m_num = len(M)
    para_num = len(drho)
    CFIM_res = np.zeros([para_num,para_num])
    for pi in range(0,m_num):
        Mp = M[pi]
        p = np.real(np.trace(np.dot(rho, Mp)))
        Cadd = np.zeros([para_num,para_num])
        if p > eps:
            for para_i in range(0,para_num):
                drho_i = drho[para_i]
                dp_i = np.real(np.trace(np.dot(drho_i,Mp)))
                for para_j in range(para_i,para_num):
                    drho_j = drho[para_j]
                    dp_j = np.real(np.trace(np.dot(drho_j,Mp)))
                    Cadd[para_i][para_j] = np.real(dp_i*dp_j/p)
                    Cadd[para_j][para_i] = np.real(dp_i*dp_j/p)
        CFIM_res += Cadd

    if para_num == 1:
        return CFIM_res[0][0]
    else:
        return CFIM_res

def SLD(rho, drho, rep="original", eps=1e-8):
    """
    Description: calculation of the symmetric logarithmic derivative (SLD)
                 for a density matrix.

    ----------
    Inputs
    ----------
    rho:
        --description: parameterized density matrix.
        --type: matrix

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, drho[0] is the derivative 
                       vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: the basis for the SLDs. 
                       rep=original means the basis for obtained SLDs is the 
                       same with the density matrix (rho).
                       rep=eigen means the SLDs are written in the eigenspace of
                       the density matrix (rho).
        --type: string {"original", "eigen"}

    ----------
    Returns
    ----------
    SLD:
        --description: SLD for the density matrix (rho).
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    SLD = [[] for i in range(0, para_num)]
        
    purity = np.trace(np.dot(rho, rho))

    if np.abs(1-purity) < eps:
        SLD_org = [[] for i in range(0, para_num)]
        for para_i in range(0, para_num):
            SLD_org[para_i] = 2*drho[para_i]

            if rep=="original":
                SLD[para_i] = SLD_org[para_i]
            elif rep=="eigen":
                val, vec = np.linalg.eig(rho)
                SLD[para_i] = np.dot(vec.conj().transpose(),np.dot(SLD_org[para_i],vec))
            else:
                raise NameError("NameError: rep should be choosen in {'original', 'eigen'}")
        if para_num == 1:
            return SLD[0]
        else:
            return SLD

    else:
        val, vec = np.linalg.eig(rho)
        for para_i in range(0, para_num):
            SLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
            for fi in range (0, dim):
                for fj in range (0, dim):
                    if np.abs(val[fi]+val[fj]) > eps:
                        SLD_eig[fi][fj] = 2*np.dot(vec[:,fi].conj().transpose(),                                                                 
                        np.dot(drho[para_i],vec[:,fj]))/(val[fi]+val[fj])
            SLD_eig[SLD_eig == np.inf] = 0.

            if rep=="original":
                SLD[para_i] = np.dot(vec,np.dot(SLD_eig,vec.conj().transpose()))
            elif rep=="eigen":
                SLD[para_i] = SLD_eig
            else:
                raise NameError("NameError: rep should be choosen in {'original', 'eigen'}")

        if para_num == 1:
            return SLD[0]
        else:
            return SLD

def RLD(rho, drho, rep="original", eps=1e-8):
    """
    Description: calculation of the right logarithmic derivative (RLD)
                 for a density matrix.

    ----------
    Inputs
    ----------
    rho:
        --description: parameterized density matrix.
        --type: matrix

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, drho[0] is the derivative 
                       vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: the basis for the RLDs. 
                       rep=original means the basis for obtained RLDs is the 
                       same with the density matrix (rho).
                       rep=eigen means the RLDs are written in the eigenspace of
                       the density matrix (rho).
        --type: string {"original", "eigen"}

    ----------
    Returns
    ----------
    RLD:
        --description: RLD for the density matrix (rho).
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    RLD = [[] for i in range(0,para_num)]
    #purity = np.trace(np.dot(rho, rho))

    val, vec = np.linalg.eig(rho)
    for para_i in range(0, para_num):
        RLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
        for fi in range (0, dim):
            for fj in range (0, dim):
                if np.abs(val[fi]) > eps:
                    RLD_eig[fi][fj] = np.dot(vec[:,fi].conj().transpose(),                                                                  np.dot(drho[para_i],vec[:,fj]))/val[fi]
        RLD_eig[RLD_eig == np.inf] = 0.

        if rep=="original":
            RLD[para_i] = np.dot(vec,np.dot(RLD_eig,vec.conj().transpose()))
        elif rep=="eigen":
            RLD[para_i] = RLD_eig
        else:
            raise NameError("NameError: rep should be choosen in {'original', 'eigen'}")
    if para_num == 1:
        return RLD[0]
    else:
        return RLD

def LLD(rho, drho, rep="original", eps=1e-8):
    """
    Description: Calculation of the left logarithmic derivative (LLD)
                for a density matrix.

    ----------
    Inputs
    ----------
    rho:
        --description: parameterized density matrix.
        --type: matrix

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, drho[0] is the derivative 
                       vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: the basis for the LLDs. 
                       rep=original means the basis for obtained LLDs is the 
                       same with the density matrix (rho).
                       rep=eigen means the LLDs are written in the eigenspace of
                       the density matrix (rho).
        --type: string {"original", "eigen"}

    ----------
    Returns
    ----------
    LLD:
        --description: LLD for the density matrix (rho).
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    LLD = [[] for i in range(0, para_num)]
    #purity = np.trace(np.dot(rho, rho))

    val, vec = np.linalg.eig(rho)
    for para_i in range(0, para_num):
        LLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
        for fi in range (0, dim):
            for fj in range (0, dim):
                if np.abs(val[fj]) > eps:
                    LLD_eig_tp = np.dot(vec[:,fi].conj().transpose(),                                                                  np.dot(drho[para_i],vec[:,fj]))/val[fj]
                    LLD_eig[fj][fi] = LLD_eig_tp.conj()
        LLD_eig[LLD_eig == np.inf] = 0.

        if rep=="original":
            LLD[para_i] = np.dot(vec,np.dot(LLD_eig,vec.conj().transpose()))
        elif rep=="eigen":
            LLD[para_i] = LLD_eig
        else:
            raise NameError("NameError: rep should be choosen in {'original', 'eigen'}")

    if para_num == 1:
        return LLD[0]
    else:
        return LLD

def QFIM(rho, drho, dtype="SLD", exportLD=False, eps=1e-8):
    """
    Description: Calculation of quantum Fisher information matrix (QFIM)
                for a density matrix.

    ----------
    Inputs
    ----------
    rho:
        --description: parameterized density matrix.
        --type: matrix

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, drho[0] is the derivative 
                       vector on the first parameter.
        --type: list (of matrix)

    dtype:
        --description: the type of logarithmic derivatives.
        --type: string {'SLD', 'RLD', 'LLD'}

    exportLD:
        --description: if True, the corresponding value of logarithmic derivatives 
                       will be exported.
        --type: bool
           
    ----------
    Returns
    ----------
    QFIM:
        --description: Quantum Fisher information matrix. If the length
                       of drho is 1, the output is a float number (QFI),
                       otherwise it returns a matrix (QFIM).
        --type: float number (QFI) or matrix (QFIM)

    """

    if type(drho) != list:
        raise TypeError('Please make sure drho is a list')

    para_num = len(drho)

    # singleparameter estimation
    if para_num == 1:
        if dtype=="SLD":
            LD_tp = SLD(rho, drho, eps=eps)
            SLD_ac = np.dot(LD_tp,LD_tp)+np.dot(LD_tp,LD_tp)
            QFIM_res = np.real(0.5*np.trace(np.dot(rho,SLD_ac)))

        elif dtype=="RLD":
            LD_tp = RLD(rho, drho, eps=eps)
            QFIM_res = np.real(np.trace(np.dot(rho,np.dot(LD_tp, LD_tp).conj().transpose())))

        elif dtype=="LLD":
            LD_tp = LLD(rho, drho, eps=eps)
            QFIM_res = np.real(np.trace(np.dot(rho,np.dot(LD_tp, LD_tp).conj().transpose())))
        else:
            raise NameError("NameError: dtype should be choosen in {'SLD', 'RLD', 'LLD'}")

    # multiparameter estimation
    else:  
        QFIM_res = np.zeros([para_num,para_num])
        if dtype=="SLD":
            LD_tp = SLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    SLD_ac = np.dot(LD_tp[para_i],LD_tp[para_j])+np.dot(LD_tp[para_j],LD_tp[para_i])
                    QFIM_res[para_i][para_j] = np.real(0.5*np.trace(np.dot(rho,SLD_ac)))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

        elif dtype=="RLD":
            LD_tp = RLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(np.trace(np.dot(rho,np.dot(LD_tp[para_i],                                                             LD_tp[para_j]).conj().transpose())))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

        elif dtype=="LLD":
            LD_tp = LLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(np.trace(np.dot(rho,np.dot(LD_tp[para_i],                                                              LD_tp[para_j]).conj().transpose())))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        else:
            raise NameError("NameError: dtype should be choosen in {'SLD', 'RLD', 'LLD'}")

    if exportLD==False:
        return QFIM_res
    else:
        return QFIM_res, LD_tp

def BCFIM(x, p, rho, drho, M=[], eps=1e-8):
    """
    Description: Calculation Bayesian version of classical Fisher information 
                 matrix (CFIM) for a density matrix.

    ---------
    Inputs
    ---------
    x:
        --description: the regimes of x for the integral.
        --type: list of arrays

    p:
        --description: the prior distribution.
        --type: multidimensional array

    rho:
        --description: parameterized density matrix.
        --type: multidimensional lists

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. 
        --type: multidimensional lists

    M:
       --description: a set of POVM. It takes the form [M1, M2, ...].
       --type: list (of matrix)

    ----------
    Returns
    ----------
    BCFIM:
        --description: Bayesian version of classical Fisher information matrix. 
                       If the length of x is one, the output is a float number (BCFI),
                       otherwise it returns a matrix (BCFIM).
        --type: float number (BCFI) or matrix (BCFIM)

    """
    para_num = len(x)
    if para_num == 1: 
        #### singleparameter senario ####
        if M==[]: 
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        p_num = len(p)
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        p_num = len(p)
        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        arr = [p[i]*F_tp[i] for i in range(p_num)]
        return simps(arr, x[0])
    else:
        #### multiparameter senario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
   
        dim = len(rho_list[0])
        if M==[]: 
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
        for i in range(len(p_list)):
            F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]

        BCFIM_res = np.zeros([para_num,para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                arr = p*F_ij
                for si in range(para_num):
                    arr = simps(arr, x[si])
                BCFIM_res[para_i][para_j] = arr
                BCFIM_res[para_j][para_i] = arr
        return BCFIM_res
    
def BQFIM(x, p, rho, drho, dtype="SLD", eps=1e-8):
    """
    Description: Calculation of Bayesian version of quantum Fisher information 
                 matrix (QFIM) for a density matrix.

    ----------
    Inputs
    ----------
    x:
        --description: the regimes of x for the integral.
        --type: list of arrays

    p:
        --description: the prior distribution.
        --type: multidimensional array

    rho:
        --description: parameterized density matrix.
        --type: multidimensional lists

    drho:
        --description: derivatives of density matrix (rho) on all parameters  
                       to be estimated. For example, 
        --type: multidimensional lists

    dtype:
        --description: the type of logarithmic derivatives.
        --type: string {'SLD', 'RLD', 'LLD'}
           
    ----------
    Returns
    ----------
    BQFIM:
        --description: Bayesian version of quantum Fisher information matrix. 
                       If the length of x is 1, the output is a float number (BQFI),
                       otherwise it returns a matrix (BQFIM).
        --type: float number (BQFI) or matrix (BQFIM)

    """

    para_num = len(x)
    if para_num == 1: 
        #### singleparameter senario ####
        p_num = len(p)
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        
        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], dtype=dtype, eps=eps)
        arr = [p[i]*F_tp[i] for i in range(p_num)]
        return simps(arr, x[0])
    else:
        #### multiparameter senario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)

        F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
        for i in range(len(p_list)):
            F_tp = QFIM(rho_list[i], drho_list[i], dtype=dtype, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]

        BQFIM_res = np.zeros([para_num,para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                arr = p*F_ij
                for si in range(para_num):
                    arr = simps(arr, x[si])
                BQFIM_res[para_i][para_j] = arr
                BQFIM_res[para_j][para_i] = arr
        return BQFIM_res

def QFIM_Bloch(r, dr, eps=1e-8):
    """
    Description: Calculation of quantum Fisher information matrix (QFIM)
                in Bloch representation.
                     
    ----------
    Inputs
    ----------
    r:
        --description: parameterized Bloch vector.
        --type: vector

    dr:
        --description: derivatives of Bloch vector on all parameters to
                        be estimated. For example, dr[0] is the derivative
                        vector on the first parameter.
        --type: list (of vector)
    """
    if type(dr) != list:
        raise TypeError('Please make sure dr is a list')

    para_num = len(dr)
    QFIM_res = np.zeros([para_num,para_num])
        
    dim = int(np.sqrt(len(r)+1))
    Lambda = suN_generator(dim)

    if dim == 2:
        #### single-qubit system ####
        r_norm = np.linalg.norm(r)**2
        if np.abs(r_norm-1.0) < eps:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num): 
                    QFIM_res[para_i][para_j] = np.real(np.inner(dr[para_i], dr[para_j]))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        else:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num): 
                    QFIM_res[para_i][para_j] = np.real(np.inner(dr[para_i], dr[para_j])\
                            +np.inner(r, dr[para_i])*np.inner(r, dr[para_j])/(1-r_norm))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
    else:
        rho = np.identity(dim, dtype=np.complex128)/dim
        for di in range(dim**2-1):
            rho += np.sqrt(dim*(dim-1)/2)*r[di]*Lambda[di]/dim
        
        G = np.zeros((dim**2-1, dim**2-1), dtype=np.complex128)
        for row_i in range(dim**2-1):
            for col_j in range(row_i, dim**2-1):
                anti_commu = np.dot(Lambda[row_i], Lambda[col_j])\
                             +np.dot(Lambda[col_j], Lambda[row_i])
                G[row_i][col_j] = 0.5*np.trace(np.dot(rho, anti_commu))
                G[col_j][row_i] = G[row_i][col_j]
        
        mat_tp = G*dim/(2*(dim-1))-np.dot(np.array(r).reshape(len(r), 1), np.array(r).reshape(1, len(r)))
        mat_inv = inv(mat_tp) 
        
        for para_m in range(0, para_num):
            for para_n in range(para_m, para_num): 
                QFIM_res[para_m][para_n] = np.real(np.dot(np.array(dr[para_n]).reshape(1, len(r)),\
                                           np.dot(mat_inv, np.array(dr[para_m]).reshape(len(r), 1)))[0][0])
                QFIM_res[para_n][para_m] = QFIM_res[para_m][para_n]
                
    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res

def QFIM_Gauss(R, dR, D, dD):
    """
    Description: Calculation of quantum Fisher information matrix (QFIM)
                for Gaussian states representation.
                     
    ----------
    Inputs
    ----------
    R:
    
    dR:
    
    D:
    
    dD:
    
    
    """

    para_num = len(dR)    
    m = int(len(R)/2)
    QFIM_res = np.zeros([para_num,para_num])
    
    C = np.array([[(D[i][j]+D[i][j])/2-R[i]*R[j] for j in range(2*m)] for i in range(2*m)])
    dC = [np.array([[(dD[k][i][j]+dD[k][i][j])/2 - dR[k][i]*R[j] - R[i]*dR[k][j] for j in range(2*m)] for i in range(2*m)]) for k in range(para_num)]
    
    C_sqrt = sqrtm(C)
    J = np.kron([[0,1],[-1,0]], np.eye(m))
    B = C_sqrt@J@C_sqrt
    P = np.eye(2*m)
    P = np.vstack([P[:][::2], P[:][1::2]])
    T, Q = schur(B)
    vals = eigvals(B)
    c = vals[::2].imag
    D = np.diagflat(c**-0.5)
    S = inv(J@C_sqrt@Q@P@np.kron([[0,1],[-1,0]], -D)).T@P.T
    
    sx = np.array([[0., 1.],[1., 0.]])
    sy = np.array([[0., -1.j],[1.j, 0.]]) 
    sz = np.array([[1., 0.],[0., -1.]])
    a_Gauss = [1j*sy, sz, np.eye(2), sx]
    
    es = [[np.eye(1,m**2,m*i+j).reshape(m,m) for j in range(m)] for i in range(m)]
    
    As = [[np.kron(s,a_Gauss[i])/np.sqrt(2) for s in es ] for i in range(4)]
    gs = [[[[np.trace(inv(S)@dC@inv(S.T)@aa.T) for aa in a] for a in A] for A in As] for dC in dC]
    G = [ np.zeros((2*m,2*m)).astype(np.longdouble) for _ in range(para_num)]
    
    for i in range(para_num):
        for j in range(m):
            for k in range(m):
                for l in range(4):
                    G[i] += np.real(gs[i][l][j][k]/(4*c[j]*c[k]+(-1)**(l+1))*inv(S.T)@As[l][j][k]@inv(S))
    
    QFIM_res += np.real([[np.trace(G[i]@dC[j])+dR[i]@inv(C)@dR[j] for j in range(para_num)] for i in range(para_num)])
    
    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res

