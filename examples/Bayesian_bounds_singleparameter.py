from quanestimation import *
import numpy as np
from scipy.integrate import simps

# initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
# free Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0_func = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
dH_func = lambda x: [0.5*B*(-sx*np.sin(x)+sz*np.cos(x))]
# prior distribution
x = [np.linspace(-0.5*np.pi, 0.5*np.pi, 100)]
mu = 0.0
eta_span = np.arange(0.2, 5.1, 0.1)
p_func = lambda x, mu, eta: np.exp(-(x-mu)**2/(2*eta**2))/(eta*np.sqrt(2*np.pi))
dp_func = lambda x, mu, eta: -(x-mu)*np.exp(-(x-mu)**2/(2*eta**2))/(eta**3*np.sqrt(2*np.pi))
# dynamics
tspan = np.linspace(0., 1.0, 1000)
f_BCRB1, f_BCRB2, f_VTB1, f_VTB2 = [], [], [], []
f_BQCRB1, f_BQCRB2, f_QVTB1, f_QVTB2, f_QZZB = [], [], [], [], []
for eta in eta_span:
    p_tp = [p_func(x[0][i], mu, eta) for i in range(len(x[0]))]
    dp_tp = [dp_func(x[0][i], mu, eta) for i in range(len(x[0]))]
    c = simps(p_tp, x[0])
    p, dp = p_tp/c, dp_tp/c

    rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x[0]))]
    drho = [[np.zeros((len(rho0), len(rho0)), dtype=np.complex128)] for i in range(len(x[0]))]
    for i in range(len(x[0])):
        H0_tp = H0_func(x[0][i])
        dH_tp = dH_func(x[0][i])
        dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[i] = rho_tp[-1]
        drho[i] = drho_tp[-1]
    f_BCRB1_tp = BCRB(x, p, rho, drho, M=[], btype=1)
    f_BCRB2_tp = BCRB(x, p, rho, drho, M=[], btype=2)
    f_VTB1_tp = VTB(x, p, dp, rho, drho, M=[], btype=1)
    f_VTB2_tp = VTB(x, p, dp, rho, drho, M=[], btype=2)
    
    f_BQCRB1_tp = BQCRB(x, p, rho, drho, btype=1)
    f_BQCRB2_tp = BQCRB(x, p, rho, drho, btype=2)
    f_QVTB1_tp = QVTB(x, p, dp, rho, drho, btype=1)
    f_QVTB2_tp = QVTB(x, p, dp, rho, drho, btype=2)
    f_QZZB_tp = QZZB(x, p, rho)
    
    f_BCRB1.append(f_BCRB1_tp)
    f_BCRB2.append(f_BCRB2_tp)
    f_VTB1.append(f_VTB1_tp)
    f_VTB2.append(f_VTB2_tp)
    
    f_BQCRB1.append(f_BQCRB1_tp)
    f_BQCRB2.append(f_BQCRB2_tp)
    f_QVTB1.append(f_QVTB1_tp)
    f_QVTB2.append(f_QVTB2_tp)
    f_QZZB.append(f_QZZB_tp)
