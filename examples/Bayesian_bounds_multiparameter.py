from quanestimation import *
import numpy as np
from scipy.integrate import simps

# initial state
rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
# free Hamiltonian
sx = np.array([[0.0, 1.0], [1.0, 0.0j]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0j], [0.0, -1.0]])
H0_func = lambda x: 0.5 * x[1] * (sx * np.cos(x[0]) + sz * np.sin(x[0]))
dH_func = lambda x: [
    0.5 * x[1] * (-sx * np.sin(x[0]) + sz * np.cos(x[0])),
    0.5 * (sx * np.cos(x[0]) + sz * np.sin(x[0])),
]
# dynamics
tspan = np.linspace(0.0, 1.0, 1000)
# prior distribution
def p_func(x, y, mu_x, mu_y, sigmax, sigmay, r):
    term1 = (
        ((x - mu_x) / sigmax) ** 2
        - 2 * r * (((x - mu_x) / sigmax)) * (((y - mu_y) / sigmay))
        + (((y - mu_y) / sigmay)) ** 2
    )
    term2 = np.exp(-term1 / 2.0 / (1 - r**2))
    term3 = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - r**2)
    return term2 / term3


def dp_func(x, y, mu_x, mu_y, sigmax, sigmay, r):
    term1 = (
        -(2 * ((x - mu_x) / sigmax**2) - 2 * r * ((y - mu_y) / sigmay) / sigmax)
        / 2.0
        / (1 - r**2)
    )
    term2 = (
        -(2 * ((y - mu_y) / sigmay**2) - 2 * r * ((x - mu_x) / sigmax) / sigmay)
        / 2.0
        / (1 - r**2)
    )
    p = p_func(x, y, mu_x, mu_y, sigmax, sigmay, r)
    return [term1 * p, term2 * p]


x = [
    np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100),
    np.linspace(0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1, 10),
]
sigmax, sigmay = 0.5, 1.0
mu_x, mu_y = 0.0, 0.0
r = 0.5
para_num = len(x)
p_tp = [[p_func(xi, xj, mu_x, mu_y, sigmax, sigmay, r) for xj in x[1]] for xi in x[0]]
dp_tp = [[dp_func(xi, xj, mu_x, mu_y, sigmax, sigmay, r) for xj in x[1]] for xi in x[0]]
c = simps(simps(p_tp, x[1]), x[0])
p = p_tp / c
dp = dp_tp / c
rho = [[[] for j in range(len(x[1]))] for i in range(len(x[0]))]
drho = [[[] for j in range(len(x[1]))] for i in range(len(x[0]))]
for i in range(len(x[0])):
    for j in range(len(x[1])):
        x_tp = [x[0][i], x[1][j]]
        H0_tp = H0_func(x_tp)
        dH_tp = dH_func(x_tp)
        dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[i][j] = rho_tp[-1]
        drho[i][j] = drho_tp[-1]
f_BCRB1 = BCRB(x, p, rho, drho, M=[], btype=1)
f_BCRB2 = BCRB(x, p, rho, drho, M=[], btype=2)
f_VTB1 = VTB(x, p, dp, rho, drho, M=[], btype=1)
f_VTB2 = VTB(x, p, dp, rho, drho, M=[], btype=2)

f_BQCRB1 = BQCRB(x, p, rho, drho, btype=1)
f_BQCRB2 = BQCRB(x, p, rho, drho, btype=2)
f_QVTB1 = QVTB(x, p, dp, rho, drho, btype=1)
f_QVTB2 = QVTB(x, p, dp, rho, drho, btype=2)
