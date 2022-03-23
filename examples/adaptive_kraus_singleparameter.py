import numpy as np
from quanestimation import *
import random

# initial state
rho0 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])

def K_func(x):
    K1 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - x[0])]])
    K2 = np.array([[0.0, np.sqrt(x[0])], [0.0, 0.0]])
    return [K1, K2]

def dK_func(x):
    dK1 = np.array([[1.0, 0.0], [0.0, -0.5 / np.sqrt(1 - x[0])]])
    dK2 = np.array([[0.0, 0.5 / np.sqrt(x[0])], [0.0, 0.0]])
    return [[dK1],[dK2]]

# measurement
M1 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]])
M = [M1, M2]

# prior distribution
x = [np.linspace(0.1, 0.9, 100)]
p = (1.0 / (x[0][-1] - x[0][0])) * np.ones(len(x[0]))
# Bayesian estimation
rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x[0]))]
for xi in range(len(x[0])):
    K_tp = K_func([x[0][xi]])
    rho[xi] = sum([K@rho0@K.conj().T for K in K_tp])
    # dK_tp = dK_func([x[0][xi]])
    # rho_tp, drho_tp = kraus(rho0, K_tp, dK_tp)
    # rho[xi] = rho_tp
# generate experimental results
np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1
pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)

# adaptive
p = pout
K, dK = AdaptiveInput(x, K_func, dK_func, channel="kraus")

apt = adaptive(x, p, rho0, max_episode=10)
apt.kraus(K, dK)
apt.CFIM(M=M, W=[], savefile=True)
