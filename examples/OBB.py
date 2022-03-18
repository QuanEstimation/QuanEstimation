import numpy as np
from quanestimation import *
from scipy.integrate import simps

#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0_func = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH_func = lambda x: [0.5*B*(-sx*np.sin(x)+sz*np.cos(x))]
d2H_func = lambda x: [0.5*B*(-sx*np.cos(x)-sz*np.sin(x))]
tspan = np.linspace(0., 1.0, 1000)
#prior distribution
x = [np.linspace(-0.5*np.pi, 0.5*np.pi, 100)]
mu, eta = 0.0, 0.5
eta_span = np.arange(0.2, 5.1, 0.1)
p_func = lambda x: np.exp(-(x-mu)**2/(2*eta**2))/(eta*np.sqrt(2*np.pi))
dp_func = lambda x: -(x-mu)*np.exp(-(x-mu)**2/(2*eta**2))/(eta**3*np.sqrt(2*np.pi))
p_tp = [p_func(x[0][i]) for i in range(len(x[0]))]
dp_tp = [dp_func(x[0][i]) for i in range(len(x[0]))]
c = simps(p_tp, x[0])
p = p_tp/c
dp = dp_tp/c
rho = [[] for i in range(len(x[0]))]
drho = [[] for i in range(len(x[0]))]
d2rho = [[] for i in range(len(x[0]))]
for i in range(len(x[0])):
    H0_tp = H0_func(x[0][i])
    dH_tp = dH_func(x[0][i])
    d2H_tp = d2H_func(x[0][i])
    dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
    rho_tp, drho_tp, d2rho_tp = dynamics.secondorder_derivative(d2H_tp)
    rho[i] = rho_tp
    drho[i] = drho_tp
    d2rho[i] = d2rho_tp
f_OBB = OBB(x, p, dp, rho, drho, d2rho)
print(f_OBB)
