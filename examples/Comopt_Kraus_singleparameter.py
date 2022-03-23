import numpy as np
from quanestimation import *
from qutip import *

# Kraus operators for the amplitude damping channel
# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])

gamma = 0.1
K1 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]])
K2 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]])
K = [K1, K2]

dK1 = np.array([[1.0, 0.0], [0.0, -0.5 / np.sqrt(1 - gamma)]])
dK2 = np.array([[0.0, 0.5 / np.sqrt(gamma)], [0.0, 0.0]])
dK = [[dK1], [dK2]]

# comprehensive optimization algorithm: PSO
PSO_paras = {
    "particle_num": 10,
    "psi0": [],
    "ctrl0": [],
    "measurement0": [],
    "max_episode": [100, 10],
    "c0": 1.0,
    "c1": 2.0,
    "c2": 2.0,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
com.kraus(K, dK)
com.SM()

# comprehensive optimization algorithm: DE
DE_paras = {
    "popsize": 10,
    "psi0": [],
    "ctrl0": [],
    "measurement0": [],
    "max_episode": 100,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
com.kraus(K, dK)
com.SM()
