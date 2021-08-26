import numpy as np
import matplotlib.pyplot as plt
from time import time
import quanestimation

T = 10.0
tnum = int(250*T)
tspan = np.linspace(0, T, tnum)
dt = tspan[1]-tspan[0]
cnum = int(tnum)

omega0 = 1.0
gamma = [0.1]

sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
sp, sm = 0.5*(sx+1.j*sy), 0.5*(sx-1.j*sy)

#initial state
psi0 = np.array([[1.+0.j],[0.+0.j]])
psi1 = np.array([[0.+0.j],[1.+0.j]])
psi_p = (psi0+psi1)/np.sqrt(2)
psi_m = (psi0-psi1)/np.sqrt(2)
rho0 = np.dot(psi_p, psi_p.conj().T)
dim = len(rho0)

#initial control coefficients
vx = 0.0*np.ones(cnum)
vy = 0.0*np.ones(cnum)
vz = 0.0*np.ones(cnum)
Hc_coeff = [vx,vy,vz]

#free Hamiltonian
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]

#control Hamiltonians
Hc_ctrl = [sx,sy,sz]

#measurement
M1 = np.dot(psi_p, psi_p.conj().transpose())
M2 = np.dot(psi_m, psi_m.conj().transpose())
M  = [M1, M2]

Lvec = [sm]
GRAPE = quanestimation.GRAPE(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, Lvec, gamma, epsilon=1e-4, max_epsides=1000)

t1 = time()
# GRAPE.QFIM(auto=False)
GRAPE.QFIM(auto=True)
print(time()-t1)