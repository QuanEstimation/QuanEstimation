import numpy as np
from quanestimation import *
from scipy import integrate
from julia import Main
import random

#initial state
rho0 = 0.5*np.array([[1.+0.0j, 1.],[1., 1.]])
#Hamiltonian
B = 0.5*np.pi
sx = np.array([[0.j, 1.],[1., 0.0]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0_res = lambda x, B: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH_res = lambda x, B: [0.5*B*(-sx*np.sin(x)+sz*np.cos(x)), 0.5*(sx*np.cos(x)+sz*np.sin(x))]

#measurement
M1 = 0.5*np.array([[1.+0.0j, 1.],[1., 1.]])
M2 = 0.5*np.array([[1.+0.0j,-1.],[-1., 1.]])
M = [M1, M2]
# M = SIC(len(rho0))

T = 1.0
tnum = int(1000*T)
tspan = np.linspace(0., T, tnum)

#### data collection ####
x = [np.linspace(-0.25*np.pi+0.1, 3.0*np.pi/4.0-0.1, 1000)]
p = (1.0/(x[0][-1]-x[0][0]))*np.ones(len(x[0]))
rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x[0]))]
H = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x[0]))]
dH = [[np.zeros((len(rho0), len(rho0)), dtype=np.complex128)] for i in range(len(x[0]))]
for xi in range(len(x[0])):
    H_tp = H0_res(x[0][xi], B)
    dH_tp = dH_res(x[0][xi], B)[0]
    dynamics = Lindblad(tspan, rho0, H_tp, [dH_tp])
    rho_tp, drho_tp = dynamics.expm()
    rho[xi] = rho_tp[-1]
    H[xi], dH[xi] = H_tp, dH_tp

#### bayesian estimation ####
random.seed(1234)
res_input = [0 for i in range(2000)]
res_rand = random.sample(range(0,len(res_input)), 500)
for i in range(len(res_rand)):
    res_input[res_rand[i]] = 1

pout, xout = Bayes(x, p, rho, M, res_input, save_file=True)
print(xout)

#### adaptive ####
p = np.load("pout.npy")[499]
apt = adaptive(x, p, tspan, rho0, H, dH, decay=[], max_episode=50, eps=1e-8)
# apt.Mopt(H0_res(0.25*np.pi, 0.5*np.pi), [dH_res(0.25*np.pi, 0.5*np.pi)])
pout, xout = apt.CFIM(M, W=[], save_file=True) 
