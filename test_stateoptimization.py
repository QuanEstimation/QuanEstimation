import numpy as np
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
dH0 = [-Lambda*np.dot(Jy, Jy)/N, -Jz]
#dissipation
decay = [[Jz,0.1]]
#measurement
M_num = len(psi0)
np.random.seed(1234)
M = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2*np.random.random(len(psi0))-np.ones(len(psi0))
    r = r_ini/np.linalg.norm(r_ini)
    phi = 2*np.pi*np.random.random(len(psi0))
    M[i] = [r[i]*np.exp(1.0j*phi[i]) for i in range(len(psi0))]
M_tp = gramschmidt(np.array(M))
Measurement = [np.dot(M_tp[i].reshape(len(psi0), 1), M_tp[i].reshape(1, len(psi0)).conj()) for i in range(M_num)]

T = 10.0
tnum = int(250*T)
tspan = np.linspace(0.0, T, tnum)

#initial psi0 for DE, PSO and NM
psi0 = [psi0]
W = np.array([[1/3,0.0],[0.0,2/3]])
# #AD algorithm
AD_paras = {'Adam':False, 'psi0':psi0, 'max_episode':500, 'epsilon':0.01, 'beta1':0.90, 'beta2':0.99}
PSO_paras = {'particle_num':10, 'psi0':psi0, 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'psi0':psi0, 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}
NM_paras = {'state_num':10, 'psi0':psi0, 'max_episode':1000, 'ar':1.0, 'ae':2.0, 'ac':0.5, 'as0':0.5, 'seed':1234}
DDPG_paras = {'layer_num':4, 'layer_dim':250, 'max_episode':500, 'seed':1234}

state = StateOpt(tspan, H0, dH0, decay, W, method='DDPG', **DDPG_paras)
state.QFIM(save_file=False)
# state.CFIM(Measurement, save_file=False)
# state.HCRB(save_file=False)