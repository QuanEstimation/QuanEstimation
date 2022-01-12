import numpy as np
import os
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
decay = [[S3,2*np.pi/cons]]
#measurement
v1 = np.array([1,0,0]).reshape(3,1)
v2 = np.array([0,1,0]).reshape(3,1)
v3 = np.array([0,0,1]).reshape(3,1)
M1 = np.dot(v1, v1.T.conj())+np.dot(v2, v2.T.conj())
M2 = np.dot(v3, v3.T.conj())

N1, N2 = 0.5*(ide2+sx/3), 0.5*(ide2-sx/3)
P1, P2 = np.kron(M1, N1), np.kron(M1, N2)
P3, P4 = np.kron(M2, N1), np.kron(M2, N2)
minput = [P1,P2,P3,P4]

T = 0.5
tnum = int(2000*T)
tspan = np.linspace(0.0, T, tnum)

AD_paras = {'Adam':False, 'measurement0':[minput], 'max_episode':500, 'epsilon':0.01, 'beta1':0.90, 'beta2':0.99, 'seed':1234}
PSO_paras = {'particle_num':10, 'measurement0':[], 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'measurement0':[], 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}
Hybrid_PSO_paras = {'particle_num':10, 'measurement0':[], 'max_episode':15, 'c0':1.0, 'c1':2.0, 'c2':2.0, 'epsilon':0.001, 'seed':1234, 'update_interval':50}
Hybrid_DE_paras = {'popsize':10, 'measurement0':[], 'max_episode':300, 'c':1.0, 'cr':0.5, 'seed':1234, 'epsilon':0.001, 'update_interval':5}

Measopt = MeasurementOpt(tspan, rho0, H0, dH0, decay, mtype='input', minput=["LC",minput,4], method='DE', **DE_paras)
Measopt.CFIM(save_file=False)
