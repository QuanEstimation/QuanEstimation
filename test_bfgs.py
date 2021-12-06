import numpy as np
from quanestimation import *
from julia import Main

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
decay = [[sp, 0.0],[sm, 0.1]]
#GRAPE 
T = 20.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)
#control coefficients
cnum = tnum-1
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]

mt, vt, epsilon, beta1, beta2, accuracy = 0.0, 0.0, 0.01, 0.90, 0.99, 1e-8
B = [np.identity(cnum), np.identity(cnum), np.identity(cnum)]
max_episode, delta_f = 300, 0.1
save_file = False

grape = Main.QuanEstimation.Gradient(H0, dH0, rho0, tspan, [sp,sm], [0.0,0.1], Hc_ctrl, Hc_coeff, [-10.0, 10.0], \
                                     np.identity(1), mt, vt, epsilon, beta1, beta2, accuracy)

Main.QuanEstimation.autoGRAPE_BFGS_QFIM(grape, save_file, max_episode, B, delta_f)
# Main.QuanEstimation.BFGS_QFIM(grape, save_file, max_episode, B)
