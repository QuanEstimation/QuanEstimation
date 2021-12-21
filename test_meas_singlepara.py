import numpy as np
from quanestimation import *

def GramSchmidt(A):
    dim = len(A)
    n = len(A[0])
    Q = [np.zeros(n, dtype=np.complex128) for i in range(dim)]
    for j in range(0, dim):
        q = A[j]
        for i in range(0, j):
            rij = np.vdot(Q[i], q)
            q = q - rij*Q[i]
        rjj = np.linalg.norm(q, ord=2)
        Q[j] = q/rjj
    return Q

#initial state
rho0 = 0.5*np.array([[1.+0.j,1.+0.j],[1.+0.j,1.+0.j]])
#Hamiltonian
omega0 = 1.0
sx = np.array([[0.+0.j, 1.+0.j],[1.+0.j, 0.+0.j]])
sy = np.array([[0.+0.j, 0.-1.j],[0.+1.j, 0.+0.j]]) 
sz = np.array([[1.+0.j, 0.+0.j],[0.+0.j, -1.+0.j]])
H0 = 0.5*omega0*sz
dH0 = [0.5*sz]
#measurement
M_num = 2
np.random.seed(1234)
M = [[] for i in range(M_num)]
for i in range(M_num):
    r_ini = 2*np.random.random(len(rho0))-np.ones(len(rho0))
    r = r_ini/np.linalg.norm(r_ini)
    phi = 2*np.pi*np.random.random(len(rho0))
    M[i] = [r[i]*np.exp(1.0j*phi[i]) for i in range(len(rho0))]
Measurement = GramSchmidt(np.array(M))
#dissipation
sp = np.array([[0.+0.j, 1.+0.j],[0.+0.j, 0.+0.j]])  
sm = np.array([[0.+0.j, 0.+0.j],[1.+0.j, 0.+0.j]]) 
decay = [[sp, 0.0],[sm, 0.1]]

AD_paras = {'Adam':False, 'max_episode':300, 'epsilon':0.01, 'beta1':0.90, 'beta2':0.99}
PSO_paras = {'particle_num':10, 'ini_particle':[], 'max_episode':[1000,100], 'c0':1.0, 'c1':2.0, 'c2':2.0, 'seed':1234}
DE_paras = {'popsize':10, 'ini_population':[], 'max_episode':1000, 'c':1.0, 'cr':0.5, 'seed':1234}

T_span = np.arange(1,41,1)
for T in T_span:

    tnum = int(250*T)
    tspan = np.linspace(0., T, tnum)

    Measopt = MeasurementOpt(tspan, rho0, H0, dH0, decay, Measurement, method='AD', **AD_paras)
    Measopt.CFIM(save_file=True)

    f_load = open('f.csv', 'r')
    f_load = ''.join([i for i in f_load])
    f_save = open("f_T%d.csv"%T,"w")
    f_save.writelines(f_load)
    f_save.close()

    m_load = open('measurements.csv', 'r')
    m_load = ''.join([i for i in m_load])
    m_save = open("measurements_T%d.csv"%T,"w")
    m_save.writelines(m_load)
    m_save.close()
