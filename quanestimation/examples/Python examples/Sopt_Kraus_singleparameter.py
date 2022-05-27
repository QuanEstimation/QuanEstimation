import numpy as np
from quanestimation import *
from qutip import *
from scipy.linalg import expm

def dU_func(s, t, H, dH):
    M1 = expm(-1.j*s*H*t)
    M2 = -1.j*dH*t
    M3 = expm(-1.j*(1-s)*H*t)
    return M1@M2@M3

def dU(dim, t, H, dH):
    S = np.linspace(0.0, 1.0, 1000)
    mat = []
    for si in S:
        mat.append(dU_func(si, t, H, dH))

    dU_tp = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            dU_tp[i][j] = np.trapz(np.array([mat[idx][i][j] for idx in range(len(S))]), S)
    return dU_tp

# LMG model
N = 10
# generation of the coherent spin state
psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
psi_css = psi_css.reshape(1, -1)[0]
# guessed state
psi0 = [psi_css]
# free Hamiltonian
Lambda, g, h = 1.0, 0.5, 0.1
Jx, Jy, Jz = jmat(0.5*N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
# derivative of the free Hamiltonian on g
dH = -Lambda*np.dot(Jy, Jy)/N
# generate Kraus operator and its derivatives
t = 10.
K = [expm(-1.j*H0*t)]
dK = [[dU(len(psi_css), t, H0, dH)]]

# State optimization algorithm: Iterative
paras = {"psi0":psi0, "max_episode": 300, "seed": 1234}
state = StateOpt(savefile=True, method="Iterative", **paras)
state.Kraus(K, dK)
state.QFIM()

