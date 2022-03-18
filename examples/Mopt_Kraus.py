import numpy as np
from quanestimation import *
from qutip import *

# Kraus operators for the amplitude damping channel
# initial state
rho0 = 0.5*np.array([[1.,1.],[1.,1.]])

gamma = 0.1
K1 = np.array([[1.0,0.0], [0.0,np.sqrt(1-gamma)]])
K2 = np.array([[0.0,0.0], [np.sqrt(gamma),0.0]])
K = [K1, K2]

dK1 = np.array([[1.0,0.0], [0.0,-0.5/np.sqrt(1-gamma)]])
dK2 = np.array([[0.0,0.0], [0.5/np.sqrt(gamma),0.0]])
dK = [[dK1, dK2]]


# measurement optimization algorithm: AD
AD_paras = {"Adam":False, "measurement0":[], "max_episode":30, "epsilon":0.001, "beta1":0.90, "beta2":0.99, "seed":1234}
Measopt = MeasurementOpt(mtype="input", minput=["LC", povm_basis, 2], savefile=False, method="AD", **AD_paras)
Measopt.dynamics(K, dK, rho0)
Measopt.CFIM()
Measopt = MeasurementOpt(mtype="input", minput=["rotation", povm_basis], savefile=False, method="AD", **AD_paras)
Measopt.CFIM()

# measurement optimization algorithm: PSO
PSO_paras = {"particle_num":10, "measurement0":[], "max_episode":[100,10], "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
Measopt = MeasurementOpt(mtype="projection", minput=[], method="PSO", **PSO_paras)
Measopt.dynamics(K, dK, rho0)
Measopt.CFIM()
Measopt = MeasurementOpt(mtype="input", minput=["LC", povm_basis, 2], savefile=False, method="PSO", **PSO_paras)
Measopt.CFIM()
Measopt = MeasurementOpt(mtype="input", minput=["rotation", povm_basis], savefile=False, method="PSO", **PSO_paras)
Measopt.CFIM()

# measurement optimization algorithm: DE
DE_paras = {"popsize":10, "measurement0":[], "max_episode":100, "c":1.0, "cr":0.5, "seed":1234}
Measopt = MeasurementOpt(mtype="projection", minput=[], method="DE", **DE_paras)
Measopt.dynamics(K, dK, rho0)
Measopt.CFIM()
Measopt = MeasurementOpt(mtype="input", minput=["LC", povm_basis, 2], savefile=False, method="DE", **DE_paras)
Measopt.CFIM()
Measopt = MeasurementOpt(mtype="input", minput=["rotation", povm_basis], savefile=False, method="DE", **DE_paras)
Measopt.CFIM()
