from quanestimation import *
import numpy as np
from qutip import spin_coherent

#### spin squeezing ####
N = 3
theta = 0.5*np.pi
phi = 0.5*np.pi
rho_CSS = spin_coherent(int(0.5*N), theta, phi, type='dm')

xi = SpinSqueezing(rho_CSS.full(), basis="Dicke", output="KU")
print(xi)

#### Target time ####
# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# free Hamiltonian
omega0 = 1.0
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
H0 = 0.5 * omega0 * sz
dH = [0.5 * sz]
# measurement
M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
M = [M1, M2]
# dissipation
sp = np.array([[0.0, 1.0], [0.0, 0.0]])
sm = np.array([[0.0, 0.0], [1.0, 0.0]])
decay = [[sp, 0.0], [sm, 0.1]]
# dynamics
tspan = np.linspace(0, 50.0, 2000)
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho = dynamics.expm()
QFI = []
for ti in range(1, 2000):
    QFI_tp = QFIM(rho[ti], drho[ti])
    QFI.append(QFI_tp)
print(QFI[242])
t = TargetTime(20.0, tspan, QFIM, rho, drho)
print(t)
