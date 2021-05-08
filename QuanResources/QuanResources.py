# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:12:12 2020

@author: JL MZ
"""
from qutip import *
from scipy.linalg import sqrtm
from numpy.linalg import eigvals

def squeezing_parameter(self，rho, N, option='Xi_S'):
    """
    Description: Calculate spin squeezing parameter for a density matrix.

    ---------
    Inputs
    ---------
    N : 
       --description: particle number.
       --type: int
    rho : 
       --description: density matrix.
       --type: matrix
       
    option:
       --description: if option=='Xi_S',the output of the squeezing parameter is defined by  
                      Kitagawa and Ueda. if option=='Xi_R',the output of the squeezing 
                      parameter is defined by Wineland et al.
       --type: string {'Xi_S', 'Xi_R'}

    ----------
    Returns
    ----------
    Xi : 
       --description: squeezing_parameter.
       --type: float

    """
    if type(rho) != Qobj:
        rho = Qobj(rho)
        
    coef = 4./float(N)
    Jx = jmat(N/2,'x')
    Jy = jmat(N/2,'y')
    Jz = jmat(N/2,'z')
    Jx_mean = np.trace(rho*Jx)
    Jy_mean = np.trace(rho*Jy)
    Jz_mean = np.trace(rho*Jz)

    costheta = Jz_mean/np.sqrt(Jx_mean**2+Jy_mean**2+Jz_mean**2)
    sintheta = np.sin(np.arccos(costheta))
    cosphi = Jx_mean/np.sqrt(Jx_mean**2+Jy_mean**2)
    if np.trace(rho*Jy) > 0:
        sinphi = np.sin(np.arccos(cosphi))
    else:
        sinphi = np.sin(2*np.pi - np.arccos(cosphi))
    Jn1 = -Jx*sinphi+Jy*cosphi
    Jn2 = -Jx*costheta*cosphi-Jy*costheta*sinphi+Jz*sintheta
    A = np.trace(rho*(Jn1*Jn1-Jn2*Jn2))
    B = np.trace(rho*(Jn1*Jn2+Jn2*Jn1))
    C = np.trace(rho*(Jn1*Jn1+Jn2*Jn2))
    
    V_minus = 0.5*(C-np.sqrt(A**2+B**2))
    V_minus = np.real(V_minus)
    Xi = coef*V_minus
    if Xi > 1.0:
        Xi=1.0
        
    if option == 'Xi_S':
        Xi = Xi
    elif option == 'Xi_R':
        Xi = (N/2)**2*Xi/(Jx_mean**2+Jy_mean**2+Jz_mean**2)
    else:
        raise NameError('NameError: option should be choosen in {Xi_S, Xi_R}')
        
    return Xi
 
def Concurrence(self，rho):
    """
    Description: Calculate the concurrence entanglement measure for a two-qubit state.
    ---------
    Inputs
    ---------
    rho : 
       --description: density matrix.
       --type: matrix

    ----------
    Returns
    ----------
    concurrence : 
       --description: concurrence.
       --type: float
    
    """
    if type(rho) != Qobj:
        rho = Qobj(rho)

    return concurrence(rho)

def Entropy_VN(self，rho):    
    """
    Description: Calculate the degree of quantum entanglement between two subsystems.
    ---------
    Inputs
    ---------
    rho : 
       --description: reduced density matrix.
       --type: matrix

    ----------
    Returns
    ----------
    Von Neumann entropy: 
       --description: Von Neumann entropy of the reduced density matrix.
       --type: float
    
    """
    eig_val = eigvals(rho)
    S = 0
    for i in range(len(eig_val)):
        if eig_val[i] !=0:
            S += -eig_val[i]*np.log(eig_val[i])
    return np.real(S)
    
    
    
    