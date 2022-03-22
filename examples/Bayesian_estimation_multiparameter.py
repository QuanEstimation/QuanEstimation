from quanestimation import *
import numpy as np
from scipy.integrate import simps
import random

# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# free Hamiltonian
x = [np.linspace(0.0, 0.5 * np.pi, 1000)]
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
H0_func = lambda x: 0.5 * x[1] * (sx * np.cos(x[0]) + sz * np.sin(x[0]))
dH_func = lambda x: [
    0.5 * x[1] * (-sx * np.sin(x[0]) + sz * np.cos(x[0])),
    0.5 * (sx * np.cos(x[0]) + sz * np.sin(x[0])),
]
# measurement
M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)
# prior distribution
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

# generate experimental results
np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1

pout, xout = Bayes(x, p, rho, y, M=M, savefile=True)
print(xout)
Lout, xout = MLE(x, rho, y, M=M, savefile=True)
print(xout)
