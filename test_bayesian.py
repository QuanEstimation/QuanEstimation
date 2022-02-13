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
Measurement = [M1, M2]

T = 1.0
tnum = int(1000*T)
tspan = np.linspace(0., T, tnum)

dim = len(rho0)
para_num = 1

#### gaussian distribution
mu = 0.0
xspan = np.linspace(-np.pi, np.pi, 100)
x1, x2 = xspan[0], xspan[-1]
sigma_span = np.arange(0.2, 2.01, 0.1)
f_BQCRB, f_TWC, f_OBB, f_QZZB = [], [], [], []
for sigma in sigma_span:
    p_gauss = lambda x: np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    dp_gauss = lambda x: -(x-mu)*np.exp(-(x-mu)**2/(2*sigma**2))/(sigma**3*np.sqrt(2*np.pi))
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
        # rho_tp, drho_tp = dynamics.expm()
        # rho, drho = rho_tp[-1], drho_tp[-1]
        rho, drho, d2rho = dynamics.secondorder_derivative([d2H])
        rho_all[i] = rho
        for j in range(para_num):
            drho_all[i][j] = drho[j]
            d2rho_all[i][j] = d2rho[j]

    f1 = BQCRB(xspan, p, rho_all, drho_all, accuracy=1e-8)
    f2 = TWCB(xspan, p, [dp], rho_all, drho_all, accuracy=1e-8)
    f3 = OBB(xspan, p, [dp], rho_all, drho_all, d2rho_all, accuracy=1e-8)
    f4 = QZZB(xspan, p, rho_all)
    f_BQCRB.append(f1)
    f_TWC.append(f2)
    f_OBB.append(f3)
    f_QZZB.append(f4)
    print(f1, f2, f3, f4)
    np.save("f_BQCRB", f_BQCRB)
    np.save("f_TWC", f_TWC)
    np.save("f_OBB", f_OBB)
    np.save("f_QZZB", f_QZZB)

#### flat distribution ####
xspan = np.linspace(0.0, np.pi, 100)
x1, x2 = xspan[0], xspan[-1]
p = (1.0/(x2-x1))*np.ones(len(xspan))
dp = np.zeros(len(xspan))

rho_all = [np.zeros((dim, dim), dtype=np.complex128) for i in range(len(xspan))]
drho_all = [[np.zeros((dim, dim), dtype=np.complex128) for j in range(para_num)] for i in range(len(xspan))]
d2rho_all = [[np.zeros((dim, dim), dtype=np.complex128) for j in range(para_num)] for i in range(len(xspan))]
for i in range(len(xspan)):
    H0 = H0_res(xspan[i])
    dH = dH_res(xspan[i])
    d2H = d2H_res(xspan[i])
    dynamics = Lindblad(tspan, rho0, H0, [dH])
    # rho_tp, drho_tp = dynamics.expm()
    # rho, drho = rho_tp[-1], drho_tp[-1]
    rho, drho, d2rho = dynamics.secondorder_derivative([d2H])
    rho_all[i] = rho
    for j in range(para_num):
        drho_all[i][j] = drho[j]
        d2rho_all[i][j] = d2rho[j]
f1 = BQCRB(xspan, p, rho_all, drho_all)
f2 = TWCB(xspan, p, [dp], rho_all, drho_all)
f3 = OBB(xspan, p, [dp], rho_all, drho_all, d2rho_all)
print(f1, f2, f3)
