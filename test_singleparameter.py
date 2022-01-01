import numpy as np
from quanestimation import *

#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]
Hc = [sx,sy,sz]
#measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
Measurement = [M1, M2]
#dissipation
sp = np.array([[0., 1.],[0., 0.]])  
sm = np.array([[0., 0.],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]]
#GRAPE 
T = 20.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)
#control coefficients
cnum = tnum
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]

GRAPE_paras = {'Adam':True, 'ctrl0':ctrl0, 'max_episode':300, 'epsilon':0.005, 'beta1':0.90, 'beta2':0.99}
PSO_paras = {'particle_num':10, 'ctrl0':ctrl0, 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'ctrl0':ctrl0, 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}
DDPG_paras = {'layer_num':4, 'layer_dim':250, 'max_episode':500, 'seed':1234}

control = ControlOpt(tspan, rho0, H0, Hc, dH0, decay, ctrl_bound=[-10.0, 10.0], method='auto-GRAPE', **GRAPE_paras)
control.QFIM(save_file=False)
# control.CFIM(Measurement, save_file=False)
