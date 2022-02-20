import numpy as np
from quanestimation import *
from scipy import integrate

def p_gauss(x, y, mu_x, mu_y, sigmax, sigmay, r):
    term1 = ((x-mu_x)/sigmax)**2-2*r*(((x-mu_x)/sigmax))*(((y-mu_y)/sigmay))+(((y-mu_y)/sigmay))**2
    term2 = np.exp(-term1/2.0/(1-r**2))
    term3 = 2*np.pi*sigmax*sigmay*np.sqrt(1-r**2)
    return term2/term3

def dxp_gauss(x, y, mu_x, mu_y, sigmax, sigmay, r):
    term = -(2*((x-mu_x)/sigmax**2)-2*r*((y-mu_y)/sigmay)/sigmax)/2.0/(1-r**2)
    return term*p_gauss(x, y, mu_x, mu_y, sigmax, sigmay, r)

def dyp_gauss(x, y, mu_x, mu_y, sigmax, sigmay, r):
    term = -(2*((y-mu_y)/sigmay**2)-2*r*((x-mu_x)/sigmax)/sigmay)/2.0/(1-r**2)
    return term*p_gauss(x, y, mu_x, mu_y, sigmax, sigmay, r)

#initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
#Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0_res = lambda x, B: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH1_res = lambda x, B: 0.5*B*(-sx*np.sin(x)+sz*np.cos(x))
dH2_res = lambda x, B: 0.5*(sx*np.cos(x)+sz*np.sin(x))

#measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
M = [M1, M2]

T = 1.0
tnum = int(1000*T)
tspan = np.linspace(0., T, tnum)

dim = len(rho0)
para_num = 2

### gaussian distribution
xspan = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
yspan = np.linspace(0.5*np.pi-0.01, 0.5*np.pi+0.01, 10)
sigmax, sigmay = 0.5, 1.0
mu_x, mu_y = 0.0, 0.0
x1, x2 = xspan[0], xspan[-1]
y1, y2 = yspan[0], yspan[-1]
rho_span = np.linspace(0.1,0.9,10)
f_BCRB1, f_BCRB2, f_VTB1, f_VTB2 = [], [], [], []
f_BQCRB1, f_BQCRB2, f_QVTB1, f_QVTB2 = [], [], [], []
f_QZZB = []
for r in rho_span:
    p_tp = [[p_gauss(xspan[j], yspan[i], mu_x, mu_y, sigmax, sigmay, r) for i in range(len(yspan))] for j in range(len(xspan))]
    dp_tp = [[[0.0, 0.0] for j in range(len(yspan))] for k in range(len(xspan))]

    for ii in range(len(xspan)):
        for jj in range(len(yspan)):
            dp_tp[ii][jj][0] = dxp_gauss(xspan[ii], yspan[jj], mu_x, mu_y, sigmax, sigmay, r)
            dp_tp[ii][jj][1] = dyp_gauss(xspan[ii], yspan[jj], mu_x, mu_y, sigmax, sigmay, r)

    S = integrate.simps(integrate.simps(p_tp, yspan), xspan)
    p = p_tp/S
    dp = dp_tp/S

    rho_all = [[np.zeros((dim, dim), dtype=np.complex128) for i in range(len(yspan))] for j in range(len(xspan))]
    drho_all = [[[np.zeros((dim, dim), dtype=np.complex128) for i in range(para_num)] for j in range(len(yspan))] for k in range(len(xspan))]
    for i in range(len(xspan)):
        for j in range(len(yspan)):
            H0 = H0_res(xspan[i], yspan[j])
            dH1 = dH1_res(xspan[i], yspan[j])
            dH2 = dH2_res(xspan[i], yspan[j])
            dynamics = Lindblad(tspan, rho0, H0, [dH1,dH2])
            rho_tp, drho_tp = dynamics.expm()
            rho, drho = rho_tp[-1], drho_tp[-1]
            rho_all[i][j] = rho
            for k in range(para_num):
                drho_all[i][j][k] = drho[k]
    f0_1 = BCRB([xspan, yspan], p, rho_all, drho_all, M=[], btype=1, eps=1e-8)
    f0_2 = BCRB([xspan,yspan], p, rho_all, drho_all, M=[], btype=2, eps=1e-8)
    f1_1 = BQCRB([xspan,yspan], p, rho_all, drho_all, btype=1, eps=1e-8)
    f1_2 = BQCRB([xspan,yspan], p, rho_all, drho_all, btype=2, eps=1e-8)
    f_BCRB1.append(f0_1)
    f_BCRB2.append(f0_2)
    f_BQCRB1.append(f1_1)
    f_BQCRB2.append(f1_2)

    f2_1 = VTB([xspan,yspan], p, dp, rho_all, drho_all, M=[], btype=1, eps=1e-8)
    f2_2 = VTB([xspan,yspan], p, dp, rho_all, drho_all, M=[], btype=2, eps=1e-8)
    f3_1 = QVTB([xspan,yspan], p, dp, rho_all, drho_all, btype=1, eps=1e-8)
    f3_2 = QVTB([xspan,yspan], p, dp, rho_all, drho_all, btype=2, eps=1e-8)
    f_VTB1.append(f2_1)
    f_VTB2.append(f2_2)
    f_QVTB1.append(f3_1)
    f_QVTB2.append(f3_2)

np.save("f_BCRB1", f_BCRB1)
np.save("f_BCRB2", f_BCRB2)
np.save("f_BQCRB1", f_BQCRB1)
np.save("f_BQCRB2", f_BQCRB2)
np.save("f_VTB1", f_VTB1)
np.save("f_VTB2", f_VTB2)
np.save("f_QVTB1", f_QVTB1)
np.save("f_QVTB2", f_QVTB2)    

#### flat distribution ####
# xspan = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
# yspan = np.linspace(0.5*np.pi-0.1, 0.5*np.pi+0.1, 10)
# x1, x2 = xspan[0], xspan[-1]
# y1, y2 = yspan[0], yspan[-1]
# p = (1.0/(x2-x1))*(1.0/(y2-y1))*np.ones((len(xspan), len(yspan)))
# dp = [[[0.0,0.0] for i in range(len(yspan))] for j in range(len(xspan))]
# rho_all = [[np.zeros((dim, dim), dtype=np.complex128) for i in range(len(yspan))] for j in range(len(xspan))]
# drho_all = [[[np.zeros((dim, dim), dtype=np.complex128) for i in range(para_num)] for j in range(len(yspan))] for k in range(len(xspan))]
# for i in range(len(xspan)):
#     for j in range(len(yspan)):
#         H0 = H0_res(xspan[i], yspan[j])
#         dH1 = dH1_res(xspan[i], yspan[j])
#         dH2 = dH2_res(xspan[i], yspan[j])
#         dynamics = Lindblad(tspan, rho0, H0, [dH1, dH2])
#         rho_tp, drho_tp = dynamics.expm()
#         rho, drho = rho_tp[-1], drho_tp[-1]
#         rho_all[i][j] = rho
#         for k in range(para_num):
#             drho_all[i][j][k] = drho[k]

# # fc = BCFIM([xspan, yspan], p, rho_all, drho_all, M=[])
# # fq = BQFIM([xspan, yspan], p, rho_all, drho_all)
# f0_1 = BCRB([xspan, yspan], p, rho_all, drho_all, M=[], btype=1, eps=1e-8)
# f0_2 = BCRB([xspan, yspan], p, rho_all, drho_all, M=[], btype=2, eps=1e-8)
# f1_1 = BQCRB([xspan, yspan], p, rho_all, drho_all, btype=1, eps=1e-8)
# f1_2 = BQCRB([xspan, yspan], p, rho_all, drho_all, btype=2, eps=1e-8)
# print(f0_1, f0_2, f1_1, f1_2)
# f2_1 = VTB([xspan, yspan], p, dp, rho_all, drho_all, M=[], btype=1, eps=1e-8)
# f2_2 = VTB([xspan, yspan], p, dp, rho_all, drho_all, M=[], btype=2, eps=1e-8)
# f3_1 = QVTB([xspan, yspan], p, dp, rho_all, drho_all, btype=1, eps=1e-8)
# f3_2 = QVTB([xspan, yspan], p, dp, rho_all, drho_all, btype=2, eps=1e-8)
# print(f2_1, f2_2, f3_1, f3_2)
