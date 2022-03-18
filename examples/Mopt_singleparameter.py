import numpy as np
from quanestimation import *

# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# Hamiltonian
omega0 = 1.0
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
H0 = 0.5 * omega0 * sz
dH = [0.5 * sz]
# measurement
M_num = 2
np.random.seed(1234)
M = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2 * np.random.random(len(rho0)) - np.ones(len(rho0))
    r = r_ini / np.linalg.norm(r_ini)
    phi = 2 * np.pi * np.random.random(len(rho0))
    M[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(len(rho0))]
Measurement = gramschmidt(np.array(M))
povm_basis = [
    np.dot(Measurement[i].reshape(-1, 1), Measurement[i].reshape(1, -1).conj())
    for i in range(M_num)
]
# dissipation
sp = np.array([[0.0, 1.0], [0.0, 0.0]])
sm = np.array([[0.0, 0.0], [1.0, 0.0]])
decay = [[sp, 0.0], [sm, 0.1]]
# dynamics
tspan = np.linspace(0.0, 20, 5000)

# projective measurement optimization
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
    mtype="input", minput=["LC", povm_basis, 2], savefile=False, method="AD", **AD_paras
)
Measopt.CFIM()
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis],
    savefile=False,
    method="AD",
    **AD_paras
)
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
Measopt = MeasurementOpt(mtype="projection", minput=[], method="PSO", **PSO_paras)
Measopt.dynamics(tspan, rho0, H0, dH, decay=decay)
Measopt.CFIM()
Measopt = MeasurementOpt(
    mtype="input",
    minput=["LC", povm_basis, 2],
    savefile=False,
    method="PSO",
    **PSO_paras
)
Measopt.CFIM()
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis],
    savefile=False,
    method="PSO",
    **PSO_paras
)
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
Measopt = MeasurementOpt(mtype="projection", minput=[], method="DE", **DE_paras)
Measopt.dynamics(tspan, rho0, H0, dH, decay=decay)
Measopt.CFIM()
Measopt = MeasurementOpt(
    mtype="input", minput=["LC", povm_basis, 2], savefile=False, method="DE", **DE_paras
)
Measopt.CFIM()
Measopt = MeasurementOpt(
    mtype="input",
    minput=["rotation", povm_basis],
    savefile=False,
    method="DE",
    **DE_paras
)
Measopt.CFIM()
