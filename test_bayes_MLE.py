import numpy as np
from quanestimation import *
from scipy.integrate import simps
import random

#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0_res = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH_res = lambda x: 0.5*B*(-sx*np.sin(x)+sz*np.cos(x))

#measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
M = [M1, M2]

T = 1.0
tnum = int(1000*T)
tspan = np.linspace(0., T, tnum)

dim = len(rho0)
para_num = 1

#### gaussian distribution
xspan = np.linspace(0.0, 0.5*np.pi, 1000)
x1, x2 = xspan[0], xspan[-1]
p = (1.0/(x2-x1))*np.ones(len(xspan))

x_true = 0.25*np.pi
H0 = H0_res(x_true)
dH = dH_res(x_true)
dynamics = Lindblad(tspan, rho0, H0, [dH])
rho_tp, drho_tp = dynamics.expm()
rho, drho = rho_tp[-1], drho_tp[-1]
for k in range(len(M)):
    p_real = np.real(np.trace(np.dot(rho, M[k])))

random.seed(1234)
res_input = [0 for i in range(5000)]
res_rand = random.sample(range(0,len(res_input)), 1250)
for i in range(len(res_rand)):
    res_input[res_rand[i]] = 1

rho = []
for j in range(len(xspan)):
    H0 = H0_res(xspan[j])
    dH = dH_res(xspan[j])
    dynamics = Lindblad(tspan, rho0, H0, [dH])
    rho_tp, drho_tp = dynamics.expm()
    rho.append(rho_tp[-1])

#### bayesian estimation ####
# max_episode = len(res_input)
# max_episode = 50
# p_out, x_out = [], []
# for i in range(max_episode):
#     pyx = np.zeros(len(xspan))
#     res_exp = res_input[i]
#     # res_exp = input("Please enter the experimental result:\n")
#     res_exp = int(res_exp)
#     for j in range(len(xspan)):
#         p_tp = np.real(np.trace(np.dot(rho[j], M[res_exp])))
#         pyx[j] = p_tp
#     arr = [pyx[m]*p[m] for m in range(len(xspan))]
#     py = simps(arr, xspan)
#     p_update = pyx*p/py
#     p_out.append(p)
#     indx = np.where(p == max(p))[0][0]
#     x_out.append(xspan[indx])

#     p = p_update

# np.save("p_out", p_out)  
# np.save("x_out", x_out)  

#### MLE ####
max_episode = len(res_input)
res = []
L = np.ones(len(xspan))
for mi in range(max_episode):
    res_exp = res_input[mi]
    for xi in range(len(xspan)):
        p_tp = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
        L[xi] = L[xi]*p_tp*2.0
    indx = np.where(L == max(L))[0][0]
    res.append(xspan[indx])

np.save("x_MLE", res)
