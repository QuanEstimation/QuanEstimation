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
dH0 = [0.5 * sz]
Hc = [sx, sy, sz]
# measurement
M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
Measurement = [M1, M2]
# dissipation
sp = np.array([[0.0, 1.0], [0.0, 0.0]])
sm = np.array([[0.0, 0.0], [1.0, 0.0]])
decay = [[sp, 0.0], [sm, 0.1]]
# GRAPE
T = 5.0
tnum = int(20 * T)
tspan = np.linspace(0.0, T, tnum)
# control coefficients
cnum = tnum
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]

GRAPE_paras = {
    "Adam": False,
    "ctrl0": ctrl0,
    "max_episode": 300,
    "epsilon": 0.01,
    "beta1": 0.90,
    "beta2": 0.99,
}
PSO_paras = {
    "particle_num": 10,
    "ctrl0": ctrl0,
    "max_episode": [1000, 100],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
DE_paras = {
    "popsize": 10,
    "ctrl0": ctrl0,
    "max_episode": 1000,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
DDPG_paras = {"layer_num": 4, "layer_dim": 250, "max_episode": 500, "seed": 1234}

control = ControlOpt(
    # method="auto-GRAPE", **GRAPE_paras
    # method="PSO", **PSO_paras
    method="DE", **DE_paras
    # method="DDPG, **DDPG_paras
)

control.dynamics(
    tspan,
    rho0,
    H0,
    dH0,
    Hc,
    decay=decay,
    ctrl_bound=[-0.2, 0.2],
)

control.mintime(2.0, method="binary", target="CFIM", M=Measurement)
