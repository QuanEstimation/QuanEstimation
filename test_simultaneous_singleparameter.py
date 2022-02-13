import numpy as np
from quanestimation import *

rho0 = 0.5*np.array([[1., 1.],[1., 1.]])

dim = 2
np.random.seed(1)
r_ini = 2*np.random.random(dim)-np.ones(dim)
r = r_ini/np.linalg.norm(r_ini)
phi = 2*np.pi*np.random.random(dim)
psi0 = [r[i]*np.exp(1.0j*phi[i]) for i in range(dim)]
psi0 = np.array(psi0)
psi0 = [psi0]

#Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0 = 0.5*omega0*sz
dH = [0.5*sz]
Hc = [sx,sy,sz]
#measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
Measurement = [M1, M2]
#dissipation
sp = np.array([[0., 1.],[0., 0.0j]])  
sm = np.array([[0., 0.0j],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]]
#GRAPE 
T = 20.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)
#control coefficients
cnum = tnum
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]

AD_paras = {"Adam":False, "psi0":psi0, "ctrl0":ctrl0, "measurement0":[], "max_episode":500, "epsilon":0.01, "beta1":0.90, "beta2":0.99, "seed":1234}
PSO_paras = {"particle_num":10, "psi0":psi0, "ctrl0":ctrl0, "measurement0":[], "max_episode":[1000,100], "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
DE_paras = {"popsize":10, "psi0":psi0, "ctrl0":ctrl0, "measurement0":[], "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}

com = ComprehensiveOpt(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-0.5, 0.5], method="DE", **DE_paras)
com.SC(target="QFIM", save_file=False)
# com.SC(target="CFIM", M=Measurement, save_file=False)
# com.CM(rho0, save_file=False)
# com.SM(save_file=False)
# com.SCM(save_file=False)
