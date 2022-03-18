from quanestimation import *
import numpy as np

# initial state
psi0 = np.array([1.0, 0.0, 0.0, 1.0])/np.sqrt(2)
rho0 = np.dot(psi0.reshape(-1,1), psi0.reshape(1,-1).conj())
# free Hamiltonian
omega1, omega2, g = 1.0, 1.0, 0.1
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
ide = np.array([[1.,0.],[0.,1.]])   
H0 = omega1*np.kron(sz,ide)+omega2*np.kron(ide,sz)+g*np.kron(sx,sx)
dH = [np.kron(ide,sz), np.kron(sx,sx)] 
# measurement
m1 = np.array([0.,0.,0.,1.])
M1 = 0.85*np.dot(m1.reshape(-1,1), m1.reshape(1,-1).conj())
M2 = 0.1*np.array([[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.]])
M = [M1, M2, np.identity(4)-M1-M2]
# dissipation
decay = [[np.kron(sz,ide), 0.05], [np.kron(ide,sz), 0.05]]
# dynamics
tspan = np.linspace(0.0, 10.0, 2000)
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho = dynamics.expm()
W = [[1.0, 0.0],[0.0, 1.0]]

F, I, f = [], [], []
for ti in range(1, 2000):
    # Cramer-Rao
    F_tp = QFIM(rho[ti], drho[ti])
    I_tp = CFIM(rho[ti], drho[ti], M)
    F.append(F_tp)
    I.append(I_tp)
    # Holevo
    f_tp, X_tp, V_tp = HCRB(rho[ti], drho[ti], W, eps=1e-6)
    f.append(f_tp)
