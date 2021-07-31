import numpy as np
from time import time
import os

from Dynamics.dynamics_AD import Lindblad_AD
from Control.GRAPE_AD import GRAPE_AD


omega0 = 1.0
gamma = [0.1]
T = 1.0
tnum = 100000
tspan = np.linspace(0, T, tnum)
dt = tspan[1]-tspan[0]
cnum = tnum
vx = 0.5*np.zeros(cnum)
vy = 0.5*np.zeros(cnum)
vz = 0.5*np.zeros(cnum)

sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])  
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
sp, sm = (sx+1.j*sy), (sx-1.j*sy)

#initial state
psi0 = np.array([[1.+0.j],[0.+0.j]])
psi1 = np.array([[0.+0.j],[1.+0.j]])
psi_p = (psi0+psi1)/np.sqrt(2)
psi_m = (psi0-psi1)/np.sqrt(2)
rho0 = np.dot(psi_p, psi_p.conj().T)
dim = len(rho0)

#time independent Hamiltonian
H0 = 0.5*omega0*sy
dH0 = [0.5*sy]

#control Hamiltonian
Hc_ctrl = [sx,sy,sz]
Hc_coeff = [vx,vy,vz]

#measurement
M1 = np.dot(psi_p, psi_p.conj().transpose())
M2 = np.dot(psi_m, psi_m.conj().transpose())
M  = [M1, M2]



Lvec = [sm]

GRAPE = GRAPE_AD(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, Lvec, gamma)

t1 = time()
GRAPE.data_generation()
t2 = time()
print(t2-t1)