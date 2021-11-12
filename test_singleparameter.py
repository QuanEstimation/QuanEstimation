import numpy as np
from quanestimation import *
from julia import Main
from time import time

#initial state
rho0 = np.array([[1.+0.j,1.+0.j],[1.+0.j,1.+0.j]])/2.0
#Hamiltonian
omega0 = 1.0
sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]
Hc_ctrl = [sx,sy,sz]
#measurement
M1 = 0.5*np.array([[1.+0.j, 1.+0.j],[1.+0.j, 1.+0.j]])
M2 = 0.5*np.array([[1.+0.j,-1.+0.j],[-1.+0.j,1.+0.j]])
M  = [M1, M2]
#dissipation
sp = np.array([[0.+0.j, 1.+0.j],[0.+0.j, 0.+0.j]])  
sm = np.array([[0.+0.j, 0.+0.j],[1.+0.j, 0.+0.j]]) 
L_opt = [sp, sm]
gamma = [0., 0.1]
#GRAPE 
T = 10.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)
#control coefficients
cnum = tnum
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]

GRAPE_paras = {'Adam':True, 'max_episodes':300, 'lr':0.01, 'beta1':0.90, 'beta2':0.99, 'mt':0.0, 'vt':0.0, 'precision':1e-6}
PSO_paras = {'particle_num':10, 'ini_particle':[], 'max_episodes':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'v0':0.1, 'seed':1234}
DE_paras = {'popsize':10, 'ini_population':[], 'max_episodes':1000, 'c':1.0, 'cr':0.5, 'u0':0.1, 'seed':1234}
DDPG_paras = {'layer_num':4, 'layer_dim':250, 'max_episodes':500, 'seed':1234}

ctrlopt = ControlOpt(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, L_opt, gamma, ctrl_bound=[-np.inf,np.inf], method='DDPG', **DDPG_paras)
ctrlopt.QFIM(save_file=True)
