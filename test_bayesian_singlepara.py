import numpy as np
from quanestimation import *
from scipy import integrate

#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0_res = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH_res = lambda x: 0.5*B*(-sx*np.sin(x)+sz*np.cos(x))
d2H_res = lambda x: 0.5*B*(-sx*np.cos(x)-sz*np.sin(x))

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
mu = 0.0
xspan = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
x1, x2 = xspan[0], xspan[-1]
eta_span = np.arange(0.2, 20.1, 0.1)
f_BCRB1, f_BCRB2, f_VTB1, f_VTB2 = [], [], [], []
f_BQCRB1, f_BQCRB2, f_QVTB1, f_QVTB2 = [], [], [], []
f_OBB, f_QZZB = [], []
for eta in eta_span:
    p_gauss = lambda x: np.exp(-(x-mu)**2/(2*eta**2))/(eta*np.sqrt(2*np.pi))
    dp_gauss = lambda x: -(x-mu)*np.exp(-(x-mu)**2/(2*eta**2))/(eta**3*np.sqrt(2*np.pi))
    p_tp = [p_gauss(xspan[i]) for i in range(len(xspan))]
    dp_tp = [dp_gauss(xspan[i]) for i in range(len(xspan))]
    S = integrate.simps(p_tp, xspan)
    p = p_tp/S
    dp = dp_tp/S

    rho_all = [np.zeros((dim, dim), dtype=np.complex128) for i in range(len(xspan))]
    drho_all = [[np.zeros((dim, dim), dtype=np.complex128) for j in range(para_num)] for i in range(len(xspan))]
    d2rho_all = [[np.zeros((dim, dim), dtype=np.complex128) for j in range(para_num)] for i in range(len(xspan))]
    for i in range(len(xspan)):
        H0 = H0_res(xspan[i])
        dH = dH_res(xspan[i])
        d2H = d2H_res(xspan[i])
        dynamics = Lindblad(tspan, rho0, H0, [dH])
        rho, drho, d2rho = dynamics.secondorder_derivative([d2H])
        rho_all[i] = rho
        for j in range(para_num):
            drho_all[i][j] = drho[j]
            d2rho_all[i][j] = d2rho[j]
    f0_1 = BCRB([xspan], p, rho_all, drho_all, M=[], btype=1, eps=1e-8)
    f0_2 = BCRB([xspan], p, rho_all, drho_all, M=[], btype=2, eps=1e-8)
    f1_1 = BQCRB([xspan], p, rho_all, drho_all, btype=1, eps=1e-8)
    f1_2 = BQCRB([xspan], p, rho_all, drho_all, btype=2, eps=1e-8)
    f_BCRB1.append(f0_1)
    f_BCRB2.append(f0_2)
    f_BQCRB1.append(f1_1)
    f_BQCRB2.append(f1_2)

    f2_1 = VTB([xspan], p, dp, rho_all, drho_all, M=[], btype=1, eps=1e-8)
    f2_2 = VTB([xspan], p, dp, rho_all, drho_all, M=[], btype=2, eps=1e-8)
    f3_1 = QVTB([xspan], p, dp, rho_all, drho_all, btype=1, eps=1e-8)
    f3_2 = QVTB([xspan], p, dp, rho_all, drho_all, btype=2, eps=1e-8)
    f_VTB1.append(f2_1)
    f_VTB2.append(f2_2)
    f_QVTB1.append(f3_1)
    f_QVTB2.append(f3_2)
    f4 = OBB([xspan], p, dp, rho_all, drho_all, d2rho_all, eps=1e-8)
    f5 = QZZB(xspan, p, rho_all)
    f_OBB.append(f4)
    f_QZZB.append(f5)

np.save("f_BCRB1", f_BCRB1)
np.save("f_BCRB2", f_BCRB2)
np.save("f_BQCRB1", f_BQCRB1)
np.save("f_BQCRB2", f_BQCRB2)
np.save("f_VTB1", f_VTB1)
np.save("f_VTB2", f_VTB2)
np.save("f_QVTB1", f_QVTB1)
np.save("f_QVTB2", f_QVTB2)    
np.save("f_OBB", f_OBB)
np.save("f_QZZB", f_QZZB)

# #### flat distribution ####
# xspan = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
# x1, x2 = xspan[0], xspan[-1]
# p = (1.0/(x2-x1))*np.ones(len(xspan))
# dp = np.zeros(len(xspan))
# para_num = 1
# rho_all = [np.zeros((dim, dim), dtype=np.complex128) for i in range(len(xspan))]
# drho_all = [[np.zeros((dim, dim), dtype=np.complex128) for i in range(para_num)] for j in range(len(xspan))]
# d2rho_all = [[np.zeros((dim, dim), dtype=np.complex128) for j in range(para_num)] for i in range(len(xspan))]
# for i in range(len(xspan)):
#     H0 = H0_res(xspan[i])
#     dH = dH_res(xspan[i])
#     dynamics = Lindblad(tspan, rho0, H0, [dH])
#     rho, drho, d2rho = dynamics.secondorder_derivative([d2H])
#     rho_all[i] = rho
#     for k in range(para_num):
#         drho_all[i][k] = drho[k]
#         d2rho_all[i][k] = d2rho[k]

# f0_1 = BCRB([xspan], p, rho_all, drho_all, M=[], btype=1, eps=1e-8)
# f0_2 = BCRB([xspan], p, rho_all, drho_all, M=[], btype=2, eps=1e-8)
# f1_1 = BQCRB([xspan], p, rho_all, drho_all, btype=1, eps=1e-8)
# f1_2 = BQCRB([xspan], p, rho_all, drho_all, btype=2, eps=1e-8)
# print(f0_1,f0_2,f1_1,f1_2)
# f2_1 = VTB([xspan], p, dp, rho_all, drho_all, M=[], btype=1, eps=1e-8)
# f2_2 = VTB([xspan], p, dp, rho_all, drho_all, M=[], btype=2, eps=1e-8)
# f3_1 = QVTB([xspan], p, dp, rho_all, drho_all, btype=1, eps=1e-8)
# f3_2 = QVTB([xspan], p, dp, rho_all, drho_all, btype=2, eps=1e-8)
# print(f2_1,f2_2,f3_1,f3_2)

# # f4 = OBB([xspan], p, dp, rho_all, drho_all, d2rho_all, eps=1e-8)
# # print(f4)
# # f5 = QZZB([xspan], p, rho_all, eps=1e-8)
# # print(f5)