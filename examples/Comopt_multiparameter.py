from quanestimation import *
import numpy as np

# initial state
rho0 = np.zeros((6, 6), dtype=np.complex128)
rho0[0][0], rho0[0][4], rho0[4][0], rho0[4][4] = 0.5, 0.5, 0.5, 0.5
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
dH = [gS * S1 + gI * I1, gS * S2 + gI * I2, gS * S3 + gI * I3]
Hc = [S1, S2, S3]
# dissipation
decay = [[S3, 2 * np.pi / cons]]
# measurement
M, C = [], []
for i in range(len(rho0)):
    M_tp = np.dot(basis(len(rho0), i), basis(len(rho0), i).conj().T)
    M.append(M_tp)
    C.append(basis(len(rho0), i).reshape(1, len(rho0))[0])
# dynamics
tspan = np.linspace(0.0, 2.0, 4000)
# initial control coefficients
cnum = 10
ctrl0 = [np.array([np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)])]

# comprehensive optimization algorithm: AD
AD_paras = {
    "Adam": False,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [np.array(C)],
    "max_episode": 10,
    "epsilon": 0.01,
    "beta1": 0.90,
    "beta2": 0.99,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="AD", **AD_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-0.2, 0.2])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(W=[], target="HCRB")

# comprehensive optimization algorithm: PSO
PSO_paras = {
    "particle_num": 10,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [np.array(C)],
    "max_episode": [100, 10],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-0.2, 0.2])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(W=[], M=M, target="CFIM")
com.SC(W=[], target="HCRB")
com.CM(rho0)
com.SM()
com.SCM()

# comprehensive optimization algorithm: DE
DE_paras = {
    "popsize": 10,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [np.array(C)],
    "max_episode": 100,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-0.2, 0.2])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(W=[], target="HCRB")
com.CM(rho0)
com.SM()
com.SCM()
