import numpy as np
from quanestimation import *

#initial state
rho0 = 0.5*np.array([[1.+0.j,1.+0.j],[1.+0.j,1.+0.j]])
#Hamiltonian
omega0 = 1.0
sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]
#measurement
M_num = 2
np.random.seed(1234)
M = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2*np.random.random(len(rho0))-np.ones(len(rho0))
    r = r_ini/np.linalg.norm(r_ini)
    phi = 2*np.pi*np.random.random(len(rho0))
    M[i] = [r[i]*np.exp(1.0j*phi[i]) for i in range(len(rho0))]
Measurement = gramschmidt(np.array(M))
#dissipation
sp = np.array([[0.+0.j, 1.+0.j],[0.+0.j, 0.+0.j]])  
sm = np.array([[0.+0.j, 0.+0.j],[1.+0.j, 0.+0.j]]) 
decay = [[sp, 0.0],[sm, 0.1]]

T = 20.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)

AD_paras = {'Adam':False, 'measurement0':[Measurement], 'max_episode':500, 'epsilon':0.001, 'beta1':0.90, 'beta2':0.99, 'seed':1234}
PSO_paras = {'particle_num':10, 'measurement0':[], 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'measurement0':[], 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}

Measopt = MeasurementOpt(tspan, rho0, H0, dH0, decay, mtype='projection', minput=[[],3], method='DE', **DE_paras)
Measopt.CFIM(save_file=False)
