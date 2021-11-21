import numpy as np
from time import time
from quanestimation import *

#initial state
rho0 = np.zeros((6,6),dtype=np.complex128)
rho0[0][0], rho0[0][4], rho0[4][0], rho0[4][4] = 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j
#Hamiltonian
sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
ide2 = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])
s1 = np.array([[0.+0.j, 1.+0.j, 0.+0.j],[1.+0.j, 0.+0.j, 1.+0.j],[0.+0.j, 1.+0.j, 0.+0.j]])/np.sqrt(2)
s2 = np.array([[0.+0.j, 0.-1.j, 0.+0.j],[0.+1.j, 0.+0.j, 0.-1.j],[0.+0.j, 0.+1.j, 0.+0.j]])/np.sqrt(2)
s3 = np.array([[1.+0.j, 0.+0.j, 0.+0.j],[0.+0.j, 0.+0.j, 0.+0.j],[0.+0.j, 0.+0.j, -1.+0.j]])
ide3 = np.array([[1.+0.j, 0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j, 1.+0.j]])
I1, I2, I3 = np.kron(ide3, sx), np.kron(ide3, sy), np.kron(ide3, sz)
S1, S2, S3 = np.kron(s1, ide2), np.kron(s2, ide2), np.kron(s3, ide2)
B1, B2, B3 = 5.0e-4, 5.0e-4, 5.0e-4
cons = 100
D = (2*np.pi*2.87*1000)/cons
gS = (2*np.pi*28.03*1000)/cons
gI = (2*np.pi*4.32)/cons
A1 = (2*np.pi*3.65)/cons
A2 = (2*np.pi*3.03)/cons
H0 = D*np.kron(np.dot(s3, s3), ide2)+gS*(B1*S1+B2*S2+B3*S3)+gI*(B1*I1+B2*I2+B3*I3)+\
     + A1*(np.kron(s1, sx)+np.kron(s2, sy)) + A2*np.kron(s3, sz)
dH1, dH2, dH3 = gS*S1+gI*I1, gS*S2+gI*I2, gS*S3+gI*I3
dH0 = [dH1, dH2, dH3]
Hc_ctrl = [S1, S2, S3]
#dissipation
L_opt = [S3]
gamma = [2*np.pi/cons]
Decay = [L_opt, gamma]

T = 2.0
tnum = int(2000*T)
tspan = np.linspace(0.0, T, tnum)
cnum = tnum
#initial control coefficients
# Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
Hc_coeff = [np.random.random(cnum), np.random.random(cnum), np.random.random(cnum)]

# initial controls for PSO and DE
ini_1, ini_2, ini_3  = np.zeros((len(Hc_ctrl), cnum)), 0.2*np.ones((len(Hc_ctrl), cnum)), -0.2*np.ones((len(Hc_ctrl), cnum))
ini_4 = np.array([np.linspace(-0.2,0.2,cnum) for i in range(len(Hc_ctrl))])
ini_5 = np.array([np.linspace(-0.2,0.0,cnum) for i in range(len(Hc_ctrl))])
ini_6 = np.array([np.linspace(0,0.2,cnum) for i in range(len(Hc_ctrl))])
ini_7 = -0.2*np.ones((len(Hc_ctrl), cnum))+0.01*np.random.random((len(Hc_ctrl), cnum))
ini_8 = -0.2*np.ones((len(Hc_ctrl), cnum))+0.01*np.random.random((len(Hc_ctrl), cnum))
ini_9 = -0.2*np.ones((len(Hc_ctrl), cnum))+0.05*np.random.random((len(Hc_ctrl), cnum))
ini_10 = -0.2*np.ones((len(Hc_ctrl), cnum))+0.05*np.random.random((len(Hc_ctrl), cnum))
ini_ctrl = [ini_1, ini_2, ini_3, ini_4, ini_5, ini_6, ini_7, ini_8, ini_9, ini_10]

GRAPE_paras = {'Adam':True, 'max_episode':300, 'lr':0.01, 'beta1':0.90, 'beta2':0.99, 'precision':1e-6}
PSO_paras = {'particle_num':10, 'ini_particle':ini_ctrl, 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'ini_population':ini_ctrl, 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}
DDPG_paras = {'layer_num':4, 'layer_dim':220, 'max_episode':500, 'seed':200}

ctrlopt = ControlOpt(tspan, rho0, H0, Hc_ctrl, dH0, Hc_coeff, Decay, ctrl_bound=[-0.1,0.1], method='DDPG', **DDPG_paras)
ctrlopt.QFIM(save_file=True)
