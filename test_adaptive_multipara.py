import numpy as np
from quanestimation import *
from scipy import integrate
from julia import Main
import random

# initial state
rho0 = 0.5*np.array([[1.+0.0j, 1.], [1., 1.]])
# Hamiltonian
B = 0.5*np.pi
sx = np.array([[0.j, 1.], [1., 0.0]])
sy = np.array([[0., -1.j], [1.j, 0.]])
sz = np.array([[1., 0.0j], [0., -1.]])


def H0_res(x, B): return 0.5*B*(sx*np.cos(x)+sz*np.sin(x))


def dH_res(x, B): return [0.5*B*(-sx*np.sin(x)+sz *
                                 np.cos(x)), 0.5*(sx*np.cos(x)+sz*np.sin(x))]


# measurement
M1 = 0.5*np.array([[1.+0.0j, 1.], [1., 1.]])
M2 = 0.5*np.array([[1.+0.0j, -1.], [-1., 1.]])
M = [M1, M2]
# M = SIC(len(rho0))

T = 1.0
tnum = int(1000*T)
tspan = np.linspace(0., T, tnum)

dim = len(rho0)


#### data collection ####
xspan = np.linspace(-0.25*np.pi+0.1, 3.0*np.pi/4.0-0.1, 100)
yspan = np.linspace(0.5*np.pi-0.1, 0.5*np.pi+0.1, 10)
x = [xspan, yspan]
p = (1.0/(xspan[-1]-xspan[0]))*(1.0/(yspan[-1]-yspan[0])) * \
    np.ones((len(xspan), len(yspan)))

rho = [[np.zeros((dim, dim), dtype=np.complex128)
        for i in range(len(yspan))] for j in range(len(xspan))]
H = [[np.zeros((dim, dim), dtype=np.complex128)
      for i in range(len(yspan))] for j in range(len(xspan))]
dH = [[[np.zeros((dim, dim), dtype=np.complex128)]
       for i in range(len(yspan))] for j in range(len(xspan))]
for xi in range(len(xspan)):
    for xj in range(len(yspan)):
        H_tp = H0_res(xspan[xi], yspan[xj])
        dH_tp = dH_res(xspan[xi], yspan[xj])
        dynamics = Lindblad(tspan, rho0, H_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[xi][xj] = rho_tp[-1]
        H[xi][xj], dH[xi][xj] = H_tp, dH_tp

## bayesian estimation ####
random.seed(1234)
res_input = [0 for i in range(500)]
res_rand = random.sample(range(0, len(res_input)), 125)
for i in range(len(res_rand)):
    res_input[res_rand[i]] = 1

pout, xout = Bayes(x, p, rho, M, res_input, save_file=True)

p = np.load("pout.npy")[499]
W = np.identity(len(rho0))

apt = adaptive(x, p, tspan, rho0, H, dH, decay=[], max_episode=20, eps=1e-8)
pout, xout = apt.CFIM(M, W=[], save_file=True)
