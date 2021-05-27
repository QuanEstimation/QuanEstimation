import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scylin
from time import time
import os
from datetime import datetime

from AsymptoticBound.CramerRao import CramerRao
from Dynamics.dynamics import Lindblad
from Control.GRAPE import GRAPE
from Control.GRAPE_without_adam import GRAPE_without_adam
from Common.common import mat_vec_convert,  dRHO

omega0 = 1.0
gamma = [0.05]
T = 5.0
tnum = 2500
tspan = np.linspace(0, T, tnum)
dt = tspan[1]-tspan[0]
cnum = tnum
vx = 0.5*np.ones(cnum)
vy = 0.5*np.ones(cnum)
vz = 0.5*np.ones(cnum)

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

#time independent Hamiltonian
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]

#control Hamiltonian
Hc_ctrl = [sx,sy,sz]
Hc_coeff = [vx,vy,vz]

#measurement
M1 = np.dot(psi_p, psi_p.conj().transpose())
M2 = np.dot(psi_m, psi_m.conj().transpose())
M  = [M1, M2]

CRB = CramerRao()

Lvec = [sz]
#GRAPE = GRAPE(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, Lvec, gamma)
GRAPE = GRAPE_without_adam(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, Lvec, gamma)