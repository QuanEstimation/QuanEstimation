import numpy as np
from quanestimation import *
from qutip import *

N = 8
# initial state
psi0 = spin_coherent(0.5 * N, 0.5 * np.pi, 0.5 * np.pi, type="ket")
psi0 = psi0.full().reshape(1, -1)[0]
# free Hamiltonian
Lambda = 1.0
g = 0.5
h = 0.1
Jx, Jy, Jz = jmat(0.5 * N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -Lambda * (np.dot(Jx, Jx) + g * np.dot(Jy, Jy)) / N - h * Jz
dH = [-Lambda * np.dot(Jy, Jy) / N]
# dissipation
decay = [[Jz, 0.1]]
# measurement
M_num = len(psi0)
np.random.seed(1234)
M_tp = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2 * np.random.random(len(psi0)) - np.ones(len(psi0))
    r = r_ini / np.linalg.norm(r_ini)
    phi = 2 * np.pi * np.random.random(len(psi0))
    M_tp[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(len(psi0))]
M_basis = gramschmidt(np.array(M_tp))
M = [
    np.dot(M_basis[i].reshape(-1, 1), M_basis[i].reshape(1, -1).conj())
    for i in range(M_num)
]
# dynamics
tspan = np.linspace(0.0, 10.0, 2500)
# initial guessed state
psi0 = [psi0]

# State optimization algorithm: AD
AD_paras = {
    "Adam": False,
    "psi0": psi0,
    "max_episode": 30,
    "epsilon": 0.01,
    "beta1": 0.90,
    "beta2": 0.99,
}
state = StateOpt(savefile=False, method="AD", **AD_paras)
state.dynamics(tspan, H0, dH, decay=decay)
state.QFIM()
state.CFIM(M=M)
state.HCRB()

# State optimization algorithm: PSO
PSO_paras = {
    "particle_num": 10,
    "psi0": psi0,
    "max_episode": [100, 10],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
state = StateOpt(savefile=False, method="PSO", **PSO_paras)
state.dynamics(tspan, H0, dH, decay=decay)
state.QFIM()
state.CFIM(M=M)
state.HCRB()

# State optimization algorithm: DE
DE_paras = {
    "popsize": 10,
    "psi0": psi0,
    "max_episode": 100,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
state = StateOpt(savefile=False, method="DE", **DE_paras)
state.dynamics(tspan, H0, dH, decay=decay)
state.QFIM()
state.CFIM(M=M)
state.HCRB()

# State optimization algorithm: DDPG
DDPG_paras = {"layer_num": 4, "layer_dim": 250, "max_episode": 50, "seed": 1234}
state = StateOpt(savefile=False, method="DDPG", **DDPG_paras)
state.dynamics(tspan, H0, dH, decay=decay)
state.QFIM()
state.CFIM(M=M)
state.HCRB()

# State optimization algorithm: NM
NM_paras = {
    "state_num": 20,
    "psi0": psi0,
    "max_episode": 100,
    "ar": 1.0,
    "ae": 2.0,
    "ac": 0.5,
    "as0": 0.5,
    "seed": 1234,
}
state = StateOpt(savefile=False, method="NM", **NM_paras)
state.dynamics(tspan, H0, dH, decay=decay)
state.QFIM()
state.CFIM(M=M)
state.HCRB()
