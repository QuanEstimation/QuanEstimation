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
dim = len(rho0)
povm_basis = []
for i in range(dim):
    M_tp = np.dot(basis(dim, i), basis(dim, i).conj().T)
    povm_basis.append(M_tp)
# dynamics
tspan = np.linspace(0.0, 2.0, 4000)

# measurement optimization algorithm: AD
AD_paras = {
    "Adam": False,
    "measurement0": [],
    "max_episode": 30,
    "epsilon": 0.001,
    "beta1": 0.90,
    "beta2": 0.99,
    "seed": 1234,
}
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis, 4],
    savefile=False,
    method="AD",
    **AD_paras
)
Measopt.dynamics(tspan, rho0, H0, dH, decay=decay)
Measopt.CFIM()

# measurement optimization algorithm: PSO
PSO_paras = {
    "particle_num": 10,
    "measurement0": [],
    "max_episode": [100, 10],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis, 4],
    savefile=False,
    method="PSO",
    **PSO_paras
)
Measopt.dynamics(tspan, rho0, H0, dH, decay=decay)
Measopt.CFIM()

# measurement optimization algorithm: DE
DE_paras = {
    "popsize": 10,
    "measurement0": [],
    "max_episode": 100,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis, 4],
    savefile=False,
    method="DE",
    **DE_paras
)
Measopt.dynamics(tspan, rho0, H0, dH, decay=decay)
Measopt.CFIM()
