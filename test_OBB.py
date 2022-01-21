import numpy as np
from quanestimation import *
from julia import Main
#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0 = 0.5*omega0*omega0*sz
dH = [omega0*sz]
d2H = [sz]
Hc = [sx,sy,sz]
#dissipation
sp = np.array([[0., 1.],[0., 0.]])  
sm = np.array([[0., 0.],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]]
T = 5.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)

x1 = 1.0
x2 = 2.0
num = 100
np.random.seed(10)
p = np.random.random(num)
p = p/np.linalg.norm(p)
dp = [np.random.random(num)]
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho, d2rho = dynamics.secondorder_derivative(d2H)
res_py = OBB(rho, drho, d2rho, x1, x2, p, dp, accuracy=1e-8)
res_jl = Main.QuanEstimation.OBB(rho, drho, d2rho, x1, x2, p, dp, accuracy=1e-8)
print(res_py)
print(res_jl)
