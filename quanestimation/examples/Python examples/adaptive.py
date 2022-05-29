from quanestimation import *
import numpy as np
import random

# initial state
rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
# free Hamiltonian
B, omega0 = 0.5 * np.pi, 1.0
sx = np.array([[0., 1.], [1., 0.]])
sy = np.array([[0., -1.j], [1.j, 0.]]) 
sz = np.array([[1., 0.], [0., -1.]])
H0_func = lambda x: 0.5*B*omega0*(sx*np.cos(x[0])+sz*np.sin(x[0]))
# derivative of free Hamiltonian in x
dH_func = lambda x: [0.5*B*omega0*(-sx*np.sin(x[0])+sz*np.cos(x[0]))]
# measurement
M1 = 0.5*np.array([[1., 1.], [1., 1.]])
M2 = 0.5*np.array([[1., -1.], [-1., 1.]])
M = [M1, M2]
# time length for the evolution
tspan = np.linspace(0., 1., 1000)
# prior distribution
x = np.linspace(-0.25*np.pi+0.1, 3.0*np.pi/4.0-0.1, 1000)
p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
# dynamics
rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for \
       i in range(len(x))]
for xi in range(len(x)):
    H_tp = H0_func([x[xi]])
    dH_tp = dH_func([x[xi]])
    dynamics = Lindblad(tspan, rho0, H_tp, dH_tp)
    rho_tp, drho_tp = dynamics.expm()
    rho[xi] = rho_tp[-1]

# # Bayesian estimation
random.seed(1234)
y = [0 for i in range(2000)]
res_rand = random.sample(range(0, len(y)), 500)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1
pout, xout = Bayes([x], p, rho, y, M=M, estimator="MAP", savefile=True)
pout = np.load("pout.npy")[499]
# generation of H and dH
H, dH = BayesInput([x], H0_func, dH_func, channel="dynamics")
# adaptive measurement
apt = Adaptive([x], pout, rho0, savefile=False, max_episode=1500, eps=1e-8)
apt.dynamics(tspan, H, dH)
apt.CFIM(M=M, W=[])
