import numpy as np
from quanestimation import *
from qutip import *

# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = np.array([[1.0,0.0], [0.0,np.sqrt(1-gamma)]])
K2 = np.array([[0.0,0.0], [np.sqrt(gamma),0.0]])
K = [K1, K2]

dK1 = np.array([[1.0,0.0], [0.0,-0.5/np.sqrt(1-gamma)]])
dK2 = np.array([[0.0,0.0], [0.5/np.sqrt(gamma),0.0]])
dK = [[dK1, dK2]]

# State optimization algorithm: AD
AD_paras = {"Adam":False, "psi0":[], "max_episode":30, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
state = StateOpt(savefile=False, method="AD", **AD_paras)
state.dynamics(K, dK)
state.QFIM()
state.CFIM()
state.HCRB()

# State optimization algorithm: PSO
PSO_paras = {"particle_num":10, "psi0":[], "max_episode":[100, 10], "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
state = StateOpt(savefile=False, method="PSO", **PSO_paras)
state.dynamics(K, dK)
state.QFIM()
state.CFIM()
state.HCRB()

# State optimization algorithm: DE
DE_paras = {"popsize":10, "psi0":[], "max_episode":100, "c":1.0, "cr":0.5, "seed":1234}
state = StateOpt(savefile=False, method="DE", **DE_paras)
state.dynamics(K, dK)
state.QFIM()
state.CFIM()
state.HCRB()

# State optimization algorithm: DDPG
DDPG_paras = {"layer_num":4, "layer_dim":250, "max_episode":50, "seed":1234}
state = StateOpt(savefile=False, method="DDPG", **DDPG_paras)
state.dynamics(K, dK)
state.QFIM()
state.CFIM()
state.HCRB()

# State optimization algorithm: NM
NM_paras = {"state_num":20, "psi0":[], "max_episode":100, "ar":1.0, "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
state = StateOpt(savefile=False, method="NM", **NM_paras)
state.dynamics(K, dK)
state.QFIM()
state.CFIM()
state.HCRB()
