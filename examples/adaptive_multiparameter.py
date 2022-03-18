import numpy as np
from quanestimation import *
import random
from itertools import product

# initial state
rho0 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
# free Hamiltonian
B = 0.5 * np.pi
sx = np.array([[0.0j, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0j], [0.0, -1.0]])
H0_func = lambda x: 0.5 * x[1] * (sx * np.cos(x[0]) + sz * np.sin(x[0]))
dH_func = lambda x: [
    0.5 * x[1] * (-sx * np.sin(x[0]) + sz * np.cos(x[0])),
    0.5 * (sx * np.cos(x[0]) + sz * np.sin(x[0])),
]
# measurement
M1 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)

# Bayesian estimation
x = [
    np.linspace(0.0, 0.5 * np.pi, 100),
    np.linspace(0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1, 10),
]
p = (
    (1.0 / (x[0][-1] - x[0][0]))
    * (1.0 / (x[1][-1] - x[1][0]))
    * np.ones((len(x[0]), len(x[1])))
)
dp = np.zeros((len(x[0]), len(x[1])))

rho = [[[] for j in range(len(x[1]))] for i in range(len(x[0]))]
for i in range(len(x[0])):
    for j in range(len(x[1])):
        x_tp = [x[0][i], x[1][j]]
        H0_tp = H0_func(x_tp)
        dH_tp = dH_func(x_tp)
        dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[i][j] = rho_tp[-1]

np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1
pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)

# adaptive
p = pout
H, dH = AdaptiveInput(x, H0_func, dH_func, channel="dynamics")
apt = adaptive(x, p, rho0, max_episode=10, eps=1e-8)
apt.dynamics(tspan, H, dH)
apt.CFIM(M=M, W=[], savefile=False)
