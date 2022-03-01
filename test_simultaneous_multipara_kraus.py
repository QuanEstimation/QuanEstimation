import numpy as np
from quanestimation import *
import scipy

# initial state
rho0 = np.zeros((6, 6), dtype=np.complex128)
rho0[0][0], rho0[0][4], rho0[4][0], rho0[4][4] = 0.5, 0.5, 0.5, 0.5

dim = len(rho0)
np.random.seed(1)
r_ini = 2 * np.random.random(dim) - np.ones(dim)
r = r_ini / np.linalg.norm(r_ini)
phi = 2 * np.pi * np.random.random(dim)
psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(dim)]
psi0 = np.array(psi0)
psi0 = [psi0]

# Hamiltonian
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
ide2 = np.array([[1.0, 0.0], [0.0, 1.0]])
s1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / np.sqrt(2)
s2 = np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]]) / np.sqrt(2)
s3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
ide3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
I1, I2, I3 = np.kron(ide3, sx), np.kron(ide3, sy), np.kron(ide3, sz)
S1, S2, S3 = np.kron(s1, ide2), np.kron(s2, ide2), np.kron(s3, ide2)
B1, B2, B3 = 5.0e-4, 5.0e-4, 5.0e-4
cons = 100
D = (2 * np.pi * 2.87 * 1000) / cons
gS = (2 * np.pi * 28.03 * 1000) / cons
gI = (2 * np.pi * 4.32) / cons
A1 = (2 * np.pi * 3.65) / cons
A2 = (2 * np.pi * 3.03) / cons
H0 = (
    D * np.kron(np.dot(s3, s3), ide2)
    + gS * (B1 * S1 + B2 * S2 + B3 * S3)
    + gI * (B1 * I1 + B2 * I2 + B3 * I3)
    + +A1 * (np.kron(s1, sx) + np.kron(s2, sy))
    + A2 * np.kron(s3, sz)
)
dH1, dH2, dH3 = gS * S1 + gI * I1, gS * S2 + gI * I2, gS * S3 + gI * I3
dH0 = [dH1, dH2, dH3]
Hc = [S1, S2, S3]

K = [scipy.linalg.expm(-1.0j * H0)]
dK = [[K @ dH0[0] for K in K], [K @ dH0[1] for K in K], [K @ dH0[2] for K in K]]


# measurement
def get_basis(dim, index):
    x = np.zeros(dim)
    x[index] = 1.0
    return x.reshape(dim, 1)


dim = len(rho0)
Measurement = []
C = []
for i in range(dim):
    M_tp = np.dot(get_basis(dim, i), get_basis(dim, i).conj().T)
    Measurement.append(M_tp)
    C.append(get_basis(dim, i).reshape(1, dim)[0])

T = 2.0
tnum = int(2000 * T)
tspan = np.linspace(0.0, T, tnum)
cnum = tnum
# initial control coefficients
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]

M0 = [np.array(C)]
AD_paras = {
    "Adam": False,
    "psi0": psi0,
    "ctrl0": ctrl0,
    "measurement0": [],
    "max_episode": 500,
    "epsilon": 0.01,
    "beta1": 0.90,
    "beta2": 0.99,
    "seed": 1234,
}
PSO_paras = {
    "particle_num": 10,
    "psi0": psi0,
    "ctrl0": ctrl0,
    "measurement0": [],
    "max_episode": [1000, 100],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
DE_paras = {
    "popsize": 10,
    "psi0": psi0,
    "ctrl0": ctrl0,
    "measurement0": M0,
    "max_episode": 1000,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}

com = ComprehensiveOpt(method="DE", **DE_paras)
# com = ComprehensiveOpt(method="PSO", **PSO_paras)
com.kraus(K, dK)
com.SM(save_file=False)
