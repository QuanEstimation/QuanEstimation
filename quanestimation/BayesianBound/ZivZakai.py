import numpy as np
from scipy.linalg import sqrtm
from scipy.integrate import simps

def trace_norm(A):

    if np.linalg.norm(A.conj().T-A) < 1e-8:
        return np.sum(np.abs(np.linalg.eigvals(A)))
    else:
        return np.sum(np.linalg.svd(A, compute_uv=False)) 

def fidelity_dm(rho, sigma):
    rho_sqrtm = sqrtm(rho)
    fidelity_sqrt = np.trace(sqrtm(np.dot(rho_sqrtm,sigma,rho_sqrtm)))
    return np.real(fidelity_sqrt)**2

def fidelity_vec(psi, phi):
    overlap = np.dot(psi.conj().T, phi)
    return conj(overlap)*overlap

def helstrom_dm(rho, sigma, P0=0.5):
    return np.real((1-trace_norm(P0*rho-(1-P0)*sigma))/2)

def helstrom_vec(psi, phi, n=1):
    return np.real((1-np.sqrt(1-fidelity_vec(psi, phi)**n))/2)

def QZZB(xspan, p, rhos):
    tau = [x - xspan[0] for x in xspan]
    N = len(xspan)
    I = simps([tau[i]*simps([2*min(p[j],p[j+i+1])*helstrom_dm(rhos[j],rhos[j+i+1]) for j in range(0,N-i-1)],xspan[0:N-i-1]) for i in range(0,N-1)],tau[0:N-1])
    
    return np.real(0.5*I)