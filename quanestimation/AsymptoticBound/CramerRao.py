
import numpy as np
import numpy.linalg as LA
from quanestimation.Common import suN_generator
#===============================================================================
#Subclass: metrology
#===============================================================================
"""
calculation of classical Fisher information matrix and quantum
Fisher information matrix.
"""
       
def CFIM(rho, drho, M, precision=1e-8):
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
       --description: derivatives of density matrix on all parameters to
                      be estimated. For example, drho[0] is the derivative
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
    if type(drho) != list or type(M) != list:
        raise TypeError('Please make sure drho and M are lists!')

    m_num = len(M)
    para_num = len(drho)
    CFIM_res = np.zeros([para_num,para_num])
    for pi in range(0,m_num):
        Mp = M[pi]
        p = np.trace(np.dot(rho,Mp))
        Cadd = np.zeros([para_num,para_num])
        if p > precision:
            for para_i in range(0,para_num):
                drho_i = drho[para_i]
                dp_i = np.trace(np.dot(drho_i,Mp))
                for para_j in range(para_i,para_num):
                    drho_j = drho[para_j]
                    dp_j = np.trace(np.dot(drho_j,Mp))
                    Cadd[para_i][para_j] = np.real(dp_i*dp_j/p)
                    Cadd[para_j][para_i] = np.real(dp_i*dp_j/p)
        CFIM_res += Cadd

    if para_num == 1:
        return CFIM_res[0][0]
    else:
        return CFIM_res

def SLD(rho, drho, rep='original', precision=1e-8):
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
        --description: derivatives of density matrix on all parameters to
                        be estimated. For example, drho[0] is the derivative
                        vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: setting the representation of SLDs. if rep=original,
                        the output of SLDs are in the original space of the
                        density matrix, if rep=eigen, then the output of SLDs
                        are in the eigen space of the density matrix.
        --type: string {'original', 'eigen'}

    ----------
    Returns
    ----------
    SLD:
        --description: SLD for a density matrix.
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError('Please make sure drho is a list!')

    para_num = len(drho)
    dim = len(rho)
    SLD = [[] for i in range(0, para_num)]
        
    purity = np.trace(np.dot(rho, rho))

    if np.abs(1-purity) < precision:
        SLD_org = [[] for i in range(0, para_num)]
        for para_i in range(0, para_num):
            SLD_org[para_i] = 2*drho[para_i]

            if rep=='original':
                SLD[para_i] = SLD_org[para_i]
            elif rep=='eigen':
                val, vec = np.linalg.eig(rho)
                SLD[para_i] = np.dot(vec.conj().transpose(),np.dot(SLD_org[para_i],vec))
            else:
                raise NameError('NameError: rep should be choosen in {original, eigen}')

    else:
        val, vec = np.linalg.eig(rho)
        for para_i in range(0, para_num):
            SLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
            for fi in range (0, dim):
                for fj in range (0, dim):
                    if np.abs(val[fi]+val[fj])> precision:
                        SLD_eig[fi][fj] = 2*np.dot(vec[:,fi].conj().transpose(),                                                                 
                        np.dot(drho[para_i],vec[:,fj]))/(val[fi]+val[fj])
            SLD_eig[SLD_eig == np.inf] = 0.

            if rep=='original':
                SLD[para_i] = np.dot(vec,np.dot(SLD_eig,vec.conj().transpose()))
            elif rep=='eigen':
                SLD[para_i] = SLD_eig
            else:
                raise NameError('NameError: rep should be choosen in {original, eigen}')
    return SLD

def RLD(rho, drho, rep='original', precision=1e-8):
    """
        Description: calculation of the right logarithmic derivative (RLD)
                    for a density matrix.

    ----------
    Inputs
    ----------
    rho:
        --description: parameterized density matrix
        --type: matrix

    drho:
        --description: derivatives of density matrix on all parameters to
                        be estimated. For example, drho[0] is the derivative
                        vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: setting the representation of SLDs. if rep=original,
                        the output of SLDs are in the original space of the
                        density matrix, if rep=eigen, then the output of SLDs
                        are in the eigen space of the density matrix.
        --type: string {'original', 'eigen'}

    ----------
    Returns
    ----------
    RLD:
        --description: RLD for a density matrix.
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError('Please make sure drho is a list!')

    para_num = len(drho)
    dim = len(rho)
    RLD = [[] for i in range(0,para_num)]
    #purity = np.trace(np.dot(rho, rho))

    val, vec = np.linalg.eig(rho)
    for para_i in range(0, para_num):
        RLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
        for fi in range (0, dim):
            for fj in range (0, dim):
                if np.abs(val[fi])> precision:
                    RLD_eig[fi][fj] = np.dot(vec[:,fi].conj().transpose(),                                                                  np.dot(drho[para_i],vec[:,fj]))/val[fi]
        RLD_eig[RLD_eig == np.inf] = 0.

        if rep=='original':
            RLD[para_i] = np.dot(vec,np.dot(RLD_eig,vec.conj().transpose()))
        elif rep=='eigen':
            RLD[para_i] = RLD_eig
        else:
            raise NameError('NameError: rep should be choosen in {original, eigen}')

    return RLD

def LLD(rho, drho, rep='original', precision=1e-8):
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
        --description: derivatives of density matrix on all parameters to
                        be estimated. For example, drho[0] is the derivative
                        vector on the first parameter.
        --type: list (of matrix)

    rep:
        --description: setting the representation of SLDs. if rep=original,
                        the output of SLDs are in the original space of the
                        density matrix, if rep=eigen, then the output of SLDs
                        are in the eigen space of the density matrix.
        --type: string {'original', 'eigen'}

    ----------
    Returns
    ----------
    LLD:
        --description: LLD for a density matrix.
        --type: list (of matrix)

    """
    if type(drho) != list:
        raise TypeError('Please make sure drho is a list!')

    para_num = len(drho)
    dim = len(rho)
    LLD = [[] for i in range(0, para_num)]
    #purity = np.trace(np.dot(rho, rho))

    val, vec = np.linalg.eig(rho)
    for para_i in range(0, para_num):
        LLD_eig = np.array([[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)])
        for fi in range (0, dim):
            for fj in range (0, dim):
                if np.abs(val[fj])> precision:
                    LLD_eig_tp = np.dot(vec[:,fi].conj().transpose(),                                                                  np.dot(drho[para_i],vec[:,fj]))/val[fj]
                    LLD_eig[fj][fi] = LLD_eig_tp.conj()
        LLD_eig[LLD_eig == np.inf] = 0.

        if rep=='original':
            LLD[para_i] = np.dot(vec,np.dot(LLD_eig,vec.conj().transpose()))
        elif rep=='eigen':
            LLD[para_i] = LLD_eig
        else:
            raise NameError('NameError: rep should be choosen in {original, eigen}')

    return LLD

def QFIM(rho, drho, dtype='SLD', rep='original', rho_type = 'DM', exportLD=False):
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
        --description: derivatives of density matrix on all parameters to
                        be estimated. For example, drho[0] is the derivative
                        vector on the first parameter.
        --type: list (of matrix)

    dtype:
        --description: setting the type used to calculate the QFI or QFIM.
        --type: string {'SLD', 'RLD', 'LLD'}

    rep:
        --description: setting the representation of LDs. if rep=original,
                        the output of LDs are in the original space of the
                        density matrix, if rep=eigen, then the output of LDs
                        are in the eigen space of the density matrix.
        --type: string {'original', 'eigen'}
        
    rho_type:
        --description: setting the type of rho.
        --type: string {'DM', 'Bloch_vector'}

       
    exportLD:
        --description: if True, for single parameter estimation senario, it will 
                        return QFI and LD.
        --type: bool
           
    ----------
    Returns
    ----------
    QFIM:
        --description: Quantum Fisher information matrix. If the length
                        of drho is one, the output is a float number (QFI),
                        otherwise it returns a matrix (QFIM).
        --type: float number (QFI) or matrix (QFIM)

    ----------
    notes
    ----------
    If the desity matrix that you input is in the original space, then the
    same with SLD, otherwise the SLD be calculated in the the eigenspace.

    """

    if type(drho) != list:
        raise TypeError('Please make sure drho is a list')

    para_num = len(drho)
    QFIM_res = np.zeros([para_num,para_num])
  
    #===========================================================
    if rho_type == 'Bloch_vector':
        r = rho
        dr = drho
        dim = int(np.sqrt(len(r)+1))
        Lambda = suN_generator(dim)
        rho = np.identity(dim)/dim
        for i in range(len(r)):
            rho = rho + np.sqrt((dim-1)/(2*dim))*r[i]*Lambda[i]

        drho = [[] for i in range(0, para_num)]
        for para in range(para_num):
            drho[para] = np.zeros((dim))
            for j in range(len(r)):
                drho[para] = drho[para] + np.sqrt((dim-1)/(2*dim))*dr[para][j]*Lambda[j]
    #=========================================================== 
        
    if dtype=='SLD':
        LD_tp = SLD(rho, drho, rep)
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                SLD_ac = np.dot(LD_tp[para_i],LD_tp[para_j])+np.dot(LD_tp[para_j],LD_tp[para_i])
                QFIM_res[para_i][para_j] = np.real(0.5*np.trace(np.dot(rho,SLD_ac)))
                QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

    elif dtype=='RLD':
        LD_tp = RLD(rho, drho, rep)
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                QFIM_res[para_i][para_j] = np.real(np.trace(np.dot(rho,np.dot(LD_tp[para_i],                                                             LD_tp[para_j]).conj().transpose())))
                QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

    elif dtype=='LLD':
        LD_tp = LLD(rho, drho, rep)
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                QFIM_res[para_i][para_j] = np.real(np.trace(np.dot(rho,np.dot(LD_tp[para_i],                                                              LD_tp[para_j]).conj().transpose())))
                QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
    else:
        raise NameError('NameError: dtype should be choosen in {SLD, RLD, LLD}')

    if para_num == 1:
        if exportLD==False:
            return QFIM_res[0][0]
        else:
            return QFIM_res[0][0], LD_tp[0]
    else:
        if exportLD==False:
            return QFIM_res
        else:
            return QFIM_res, LD_tp
            