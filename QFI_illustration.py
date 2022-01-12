import numpy as np
from quanestimation import *
#initial state
rho0 = 0.5*np.array([[1.,1.],
                     [1.,1.]])
#free Hamiltonian
omega0 = 1.0
sz = np.array([[1., 0.],
               [0., -1.]])
H0 = 0.5*omega0*sz
dH = [0.5*sz]
#measurement
M1 = 0.5*np.array([[1.,1.],
                   [1.,1.]])
M2 = 0.5*np.array([[1.,-1.],
                   [-1.,1.]])
M = [M1, M2]
#dissipation
sp = np.array([[0., 1.],
               [0., 0.]])
sm = np.array([[0., 0.],
               [1., 0.]])
decay = [[sp, 0.0],[sm, 0.1]]
#dynamics
tspan = np.linspace(0, 50.0, 2000)
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho = dynamics.expm()
#calculation of QFI and CFI
QFI, CFI = [], []
for ti in range(1,2000):
    QFI_tp = QFIM(rho[ti], drho[ti])
    CFI_tp = CFIM(rho[ti], drho[ti], M)
    QFI.append(QFI_tp)
    CFI.append(CFI_tp)
