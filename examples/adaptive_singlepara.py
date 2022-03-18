import numpy as np
from quanestimation import *
import random

# initial state
rho0 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
# free Hamiltonian
B = 0.5 * np.pi
sx = np.array([[0.0j, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0j], [0.0, -1.0]])
H0_func = lambda x: 0.5 * B * (sx * np.cos(x[0]) + sz * np.sin(x[0]))
dH_func = lambda x: [0.5 * B * (-sx * np.sin(x[0]) + sz * np.cos(x[0]))]
# measurement
M1 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)
# prior distribution
x = [np.linspace(-0.25 * np.pi + 0.1, 3.0 * np.pi / 4.0 - 0.1, 100)]
p = (1.0 / (x[0][-1] - x[0][0])) * np.ones(len(x[0]))
# Bayesian estimation
rho = [np.zeros((len(rho0), len(rho0)), LDtype=np.complex128) for i in range(len(x[0]))]
for xi in range(len(x[0])):
    H_tp = H0_func([x[0][xi]])
    dH_tp = dH_func([x[0][xi]])
    dynamics = Lindblad(tspan, rho0, H_tp, dH_tp)
    rho_tp, drho_tp = dynamics.expm()
    rho[xi] = rho_tp[-1]
# generate experimental results
np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1
pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)

# adaptive
p = pout
H, dH = AdaptiveInput(x, H0_func, dH_func, channel="dynamics")

apt = adaptive(x, p, rho0, max_episode=10)
apt.dynamics(tspan, H, dH)
apt.CFIM(M=M, W=[], savefile=False)
