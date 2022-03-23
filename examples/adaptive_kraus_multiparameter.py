import numpy as np
from quanestimation import *
import random

# initial state
rho0 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])

psi0 = np.array([[1],[0]])
psi1 = np.array([[0],[1]])

def K_func(x):
    n, p = x[0], x[1]
    K0 = np.sqrt(1-n)*(psi0@psi0.T+np.sqrt(1-p)*psi1@psi1.T)
    K1 = np.sqrt(p-p*n)*psi0@psi1.T
    K2 = np.sqrt(n)*(np.sqrt(1-p)*psi0@psi0.T+psi1@psi1.T)
    K3 = np.sqrt(p*n)*psi1@psi0.T
    return [K0, K1, K2, K3]

def dK_func(x):
    n, p = x[0], x[1]
    dK0_n = -0.5*(psi0@psi0.T+np.sqrt(1-p)*psi1@psi1.T)/np.sqrt(1-n)
    dK1_n = -0.5*p*psi0@psi1.T/np.sqrt(p-p*n)
    dK2_n = 0.5*(np.sqrt(1-p)*psi0@psi0.T+psi1@psi1.T)/np.sqrt(n)
    dK3_n = 0.5*p*psi1@psi0.T/np.sqrt(p*n)
    dK0_p = -0.5*np.sqrt(1-n)*psi1@psi1.T/np.sqrt(1-p)
    dK1_p = 0.5*(1-n)*psi0@psi1.T/np.sqrt(p-p*n)
    dK2_p = -0.5*np.sqrt(n)*psi0@psi0.T/np.sqrt(1-p)
    dK3_p = -0.5*np.sqrt(n)*psi0@psi0.T/np.sqrt(1-p)
    dK3_p = 0.5*n*psi1@psi0.T/np.sqrt(p*n)
    return [[dK0_n, dK0_p], [dK1_n, dK1_p], [dK2_n, dK2_p], [dK3_n, dK3_p]]

# measurement
M1 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)

#Bayesian estimation
x = [np.linspace(0.1, 0.9, 100), np.linspace(0.1, 0.9, 10)]
p = ((1.0 / (x[0][-1] - x[0][0])) * (1.0 / (x[1][-1] - x[1][0]))* np.ones((len(x[0]), len(x[1]))))
dp = np.zeros((len(x[0]), len(x[1])))

rho = [[[] for j in range(len(x[1]))] for i in range(len(x[0]))]
for i in range(len(x[0])):
    for j in range(len(x[1])):
        x_tp = [x[0][i], x[1][j]]
        K_tp = K_func(x_tp)
        dK_tp = dK_func(x_tp)
        rho_tp, drho_tp = kraus(rho0, K_tp, dK_tp)
        rho[i][j] = rho_tp

np.random.seed(1234)
y = [0 for i in range(500)]
res_rand = random.sample(range(0, len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1
pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)

# adaptive
p = pout
K, dK= AdaptiveInput(x, K_func, dK_func, channel="kraus")
apt = adaptive(x, p, rho0, max_episode=10, eps=1e-8)
apt.kraus(K, dK)
apt.CFIM(M=M, W=[], savefile=False)
