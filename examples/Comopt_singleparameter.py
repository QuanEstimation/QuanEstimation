from quanestimation import *
import numpy as np

rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# Hamiltonian
omega0 = 1.0
sx = np.array([[0.0, 1.0], [1.0, 0.0j]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0j], [0.0, -1.0]])
H0 = 0.5 * omega0 * sz
dH = [0.5 * sz]
Hc = [sx, sy, sz]
# measurement
M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dissipation
sp = np.array([[0.0, 1.0], [0.0, 0.0j]])
sm = np.array([[0.0, 0.0j], [1.0, 0.0]])
decay = [[sp, 0.0], [sm, 0.1]]
# GRAPE
tspan = np.linspace(0.0, 20.0, 5000)
# control coefficients
cnum = len(tspan) - 1
ctrl0 = [np.array([np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)])]

# comprehensive optimization algorithm: AD
AD_paras = {
    "Adam": False,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [],
    "max_episode": 10,
    "epsilon": 0.01,
    "beta1": 0.90,
    "beta2": 0.99,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="AD", **AD_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(M=M, W=[], target="CFIM")
com.SC(W=[], target="HCRB")

# comprehensive optimization algorithm: PSO
PSO_paras = {
    "particle_num": 10,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [],
    "max_episode": [100, 10],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(W=[], target="HCRB")
com.CM(rho0)
com.SM()
com.SCM()

# comprehensive optimization algorithm: DE
DE_paras = {
    "popsize": 10,
    "psi0": [],
    "ctrl0": ctrl0,
    "measurement0": [],
    "max_episode": 100,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
com.SC(W=[], target="QFIM", LDtype="SLD")
com.SC(W=[], M=M, target="CFIM")
com.SC(W=[], target="HCRB")
com.CM(rho0)
com.SM()
com.SCM()
