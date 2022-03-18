from quanestimation import *
import numpy as np
from scipy.integrate import simps
import random

# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# free Hamiltonian
B = np.pi / 2.0
x = [np.linspace(0.0, 0.5 * np.pi, 1000)]
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
H0_func = lambda x: 0.5 * B * (sx * np.cos(x) + sz * np.sin(x))
dH_func = lambda x: [0.5 * B * (-sx * np.sin(x) + sz * np.cos(x))]
# measurement
M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)
# prior distribution
x = [np.linspace(0.0, 0.5 * np.pi, 100)]
p = (1.0 / (x[0][-1] - x[0][0])) * np.ones(len(x[0]))

rho = [[] for i in range(len(x[0]))]
for i in range(len(x[0])):
    H0_tp = H0_func(x[0][i])
    dH_tp = dH_func(x[0][i])
    dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
    rho_tp, drho_tp = dynamics.expm()
    rho[i] = rho_tp[-1]
# generate experimental results
np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1

pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)
print(xout)
Lout, xout = MLE(x, rho, y, M=M, savefile=False)
print(xout)
