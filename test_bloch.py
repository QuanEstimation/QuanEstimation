import numpy as np
from quanestimation import *
from julia import Main

#initial state
# rho0 = 0.5*np.array([[1.+0.j, 1.],[1., 1.]])
rho0 = 0.25*np.array([[1.+0.j,1.+0.j,1.+0.j,1.+0.j],[1.+0.j,1.+0.j,1.+0.j,1.+0.j],\
                      [1.+0.j,1.+0.j,1.+0.j,1.+0.j],[1.+0.j,1.+0.j,1.+0.j,1.+0.j]])
#Hamiltonian
omega1, omega2, g = 1.0, 1.2, 0.1
ide = np.array([[1.+0.j,0.],[0.,1.]])  
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., 0.-1.j],[0.+1.j, 0.]]) 
sz = np.array([[1.+0.j, 0.],[0., -1.]])
# H0 = omega1*sz
H0 = omega1*np.kron(sz,ide)+omega2*np.kron(ide,sz)+g*np.kron(sz,sz)
# dH0 = [sz] 
dH0 = [np.kron(sz,ide), np.kron(ide,sz)] 
# dynamics
T = 1.0
tnum = int(500*T)
tspan = np.linspace(0., T, tnum)

para_num = len(dH0)
dim = len(rho0)
Lambda = suN_generator(dim)
dynamics = Lindblad(tspan, rho0, H0, dH0)
rho_tp, drho_tp = dynamics.expm()
F, F_bloch_py, F_bloch_jl = [], [], []
for i in range(tnum):
    F.append(QFIM(rho_tp[i], drho_tp[i]))
    r = np.array([np.sqrt(dim/(2*(dim-1)))*np.trace(np.dot(rho_tp[i], Lambda[j])) for j in range(len(Lambda))])
    dr = [np.zeros(len(Lambda)) for k in range(para_num)]
    for m in range(para_num):
        dr[m] = np.array([np.sqrt(dim/(2*(dim-1)))*np.trace(np.dot(drho_tp[i][m], Lambda[j])) for j in range(len(Lambda))])

    F_bloch_py.append(QFIM_Bloch(r, dr))
    F_bloch_tp = Main.QuanEstimation.QFIM_Bloch(r,dr)
    F_bloch_jl.append(F_bloch_tp)

print(np.array(F)-np.array(F_bloch_jl))
print(np.array(F)-np.array(F_bloch_py))
