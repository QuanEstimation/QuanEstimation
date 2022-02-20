import numpy as np
from quanestimation import *
import os

#initial state
rho0 = 0.5*np.array([[1.,1.],[1.,1.]])

dim = len(rho0)
np.random.seed(1)
r_ini = 2*np.random.random(dim)-np.ones(dim)
r = r_ini/np.linalg.norm(r_ini)
phi = 2*np.pi*np.random.random(dim)
psi0 = [r[i]*np.exp(1.0j*phi[i]) for i in range(dim)]
psi0 = np.array(psi0)
psi0 = [psi0]

#Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0 = 0.5*omega0*sz
dH = [0.5*sz]
Hc = [sx,sy,sz]
#dissipation
sp = np.array([[0., 1.],[0., 0.]])  
sm = np.array([[0., 0.],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]] 
#measurement
M_num = 2
np.random.seed(1234)
M = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2*np.random.random(len(rho0))-np.ones(len(rho0))
    r = r_ini/np.linalg.norm(r_ini)
    phi = 2*np.pi*np.random.random(len(rho0))
    M[i] = [r[i]*np.exp(1.0j*phi[i]) for i in range(len(rho0))]
Measurement = gramschmidt(np.array(M))
povm_basis = [np.dot(Measurement[i].reshape(len(rho0), 1), Measurement[i].reshape(1, len(rho0)).conj()) for i in range(M_num)]

T = 20.0
tnum = int(250*T)
tspan = np.linspace(0., T, tnum)

# initial control coefficients
cnum = tnum
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]
ctrl_opt = Hc_coeff
psi_opt = psi0

episodes = 5
for ei in range(episodes):
    # state optimization
    DE_paras = {"popsize":10, "psi0":psi_opt, "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
    state = StateOpt(tspan, H0, dH, Hc=Hc, ctrl=ctrl_opt, decay=decay, method="DE", **DE_paras)
    state.QFIM(save_file=True)
    ####  load f and rename ####
    f_load = open('f.csv', 'r')
    f_load = ''.join([i for i in f_load])
    f_save = open("f_Sopt%d.csv"%ei,"w")
    f_save.writelines(f_load)
    f_save.close()

    s_load = open('states.csv', 'r')
    s_load = ''.join([i for i in s_load])
    s_save = open("states_Sopt%d.csv"%ei,"w")
    s_save.writelines(s_load)
    s_save.close()
    if os.path.exists('f.csv'):
        os.remove('f.csv')

    # control optimization
    psi_save = np.genfromtxt("states.csv", dtype=np.complex128)
    csv2npy_states(psi_save)
    psi_opt = np.load("states.npy")
    psi_opt = psi_opt.reshape(1, len(rho0))[0]
    rho_opt = np.dot(psi_opt.reshape(len(rho0), 1), psi_opt.reshape(1, len(rho0)).conj())
    psi_opt = [psi_opt]

    DE_paras = {"popsize":10, "ctrl0":ctrl0, "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
    control = ControlOpt(tspan, rho_opt, H0, dH, Hc, decay=decay, ctrl_bound=[-0.5, 0.5], method="DE", **DE_paras)
    control.QFIM(save_file=True)
    f_load = open('f.csv', 'r')
    f_load = ''.join([i for i in f_load])
    f_save = open("f_Copt%d.csv"%ei,"w")
    f_save.writelines(f_load)
    f_save.close()

    c_load = open('controls.csv', 'r')
    c_load = ''.join([i for i in c_load])
    c_save = open("controls_Copt%d.csv"%ei,"w")
    c_save.writelines(c_load)
    c_save.close()
    if os.path.exists('f.csv'):
        os.remove('f.csv')
    
    ctrl_save = np.genfromtxt("controls.csv")
    csv2npy_controls(ctrl_save, len(Hc))
    ctrl_opt = np.load("controls.npy")[0]
    ctrl_opt = [ctrl_opt[i] for i in range(len(Hc))]
    ctrl0 = [np.array(ctrl_opt)]

# # measurement optimization
psi_save = np.genfromtxt("states.csv", dtype=np.complex128)
csv2npy_states(psi_save)
psi_opt = np.load("states.npy")
rho_opt = np.dot(psi_opt.reshape(len(rho0), 1), psi_opt.reshape(1, len(rho0)).conj())

ctrl_save = np.genfromtxt("controls.csv")
csv2npy_controls(ctrl_save, len(Hc))
ctrl_opt = np.load("controls.npy")[0]

DE_paras = {"popsize":10, "measurement0":[], "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
Measopt = MeasurementOpt(tspan, rho_opt, H0, dH, Hc=Hc, ctrl=ctrl_opt, decay=decay, mtype="projection", minput=[], method="DE", **DE_paras)
Measopt.CFIM(save_file=True)
