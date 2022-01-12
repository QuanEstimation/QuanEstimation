import numpy as np
from quanestimation import *
from julia import Main

#initial state
psi0 = np.array([1.0, 0.0, 0.0, 1.0])/np.sqrt(2)
rho0 = np.dot(psi0.reshape(4,1), psi0.reshape(1,4).conj())

#Hamiltonian
omega1, omega2, g = 1.0, 1.0, 0.1
ide = np.array([[1.+0.j,0.+0.j],[0.+0.j,1.+0.j]])   
sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
H0 = omega1*np.kron(sz,ide)+omega2*np.kron(ide,sz)+g*np.kron(sx,sx)
dH0 = [np.kron(ide,sz), np.kron(sx,sx)] 

#measurement
m1 = np.array([0,0,0,1.0])
M1 = 0.85*np.dot(m1.reshape(4,1), m1.reshape(1,4).conj())
M2 = 0.1*np.array([[1.+0.j,1.+0.j,1.+0.j,1.+0.j],[1.+0.j,1.+0.j,1.+0.j,1.+0.j],\
                    [1.+0.j,1.+0.j,1.+0.j,1.+0.j],[1.+0.j,1.+0.j,1.+0.j,1.+0.j]])
M = [M1, M2, np.identity(4)-M1-M2]

#dissipation
decay = [[np.kron(sz,ide), 0.05], [np.kron(ide,sz), 0.05]]

#GRAPE 
T = 10.0
tnum = int(100*T)
tspan = np.linspace(0, T, tnum)

dynamics = Lindblad(tspan, rho0, H0, dH0, decay)
rho, drho = dynamics.expm()
W = [[1.0, 0.0],[0.0, 1.0]]

QFI, CFI = [], []
HCRB_jl, HCRB_py = [], []
for rho_i, drho_i in zip(rho, drho):
    Cramer-Rao
    QFI_tp = QFIM(rho_i, drho_i)
    CFI_tp = CFIM(rho_i, drho_i, M)
    QFI.append(QFI_tp)
    CFI.append(CFI_tp)

    # Holevo
    f_jl, X_jl, V_jl = Main.QuanEstimation.Holevo_bound(rho_i, drho_i, W, 1e-8) 
    f_py, X_py, V_py = Holevo_bound(rho_i, drho_i, W, 1e-6)
    HCRB_jl.append(f_jl)
    HCRB_py.append(f_py)
