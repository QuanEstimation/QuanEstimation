import numpy as np
from time import time
from quanestimation import *
from qutip import *

N = 8
#initial state
psi0 = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type='ket')
psi0 = psi0.full().reshape(1, len(psi0.full()))[0]
#Hamiltonian
Lambda = 1.0
g = 0.5
h = 0.1
Jx, Jy, Jz = jmat(0.5*N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -Lambda*(np.dot(Jx, Jx)+g*np.dot(Jy, Jy))/N-h*Jz
dH0 = [-Lambda*np.dot(Jy, Jy)/N]
#dissipation
L_opt = [Jz]
gamma = [0.1]

T = 10.0
tnum = int(200*T)
tspan = np.linspace(0.0, T, tnum)

#initial psi0 for DE, PSO and NM
ini_state = [psi0]
# W = np.array([[0.5,0.0],[0.0,0.5]])
# #AD algorithm
AD_paras = {'lr':0.01, 'epsilon':1e-8, 'max_episodes':300, 'Adam':False}
PSO_paras = {'particle_num':10, 'ini_particle':ini_state, 'c0':0.5, 'c1':0.1, 'c2':0.6, 'v0':0.01, 'seed':1234, 'max_episodes':[1000,100]}
DE_paras = {'popsize':10, 'ini_population':ini_state, 'c':0.5, 'cr':0.5, 'seed':1234, 'max_episodes':1000}
NM_paras = {'state_num':10, 'ini_state':ini_state, 'a_r':1.2, 'a_e':2.5, 'a_c':0.7, 'a_s':0.7, 'seed':200, 'epsilon':1e-6, 'max_episodes':1000}

stateopt = StateOpt(tspan, psi0, H0, dH=dH0, Liouville_operator=L_opt, gamma=gamma, method='DE', **DE_paras)
stateopt.QFIM(save_file=True)
