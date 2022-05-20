from quanestimation import *
import numpy as np

# the number of photons
N = 8
# probe state
psi = np.zeros((N+1)**2).reshape(-1, 1)
for k in range(N+1):
    psi += np.sin((k+1)*np.pi/(N+2))* \
           np.kron(basis(N+1, k), basis(N+1, N-k))
psi = np.sqrt(2/(2+N))*psi
rho0 = np.dot(psi, psi.conj().T)
# prior distribution
x = np.linspace(-np.pi, np.pi, 100)
p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
apt = adapt_MZI(x, p, rho0)
apt.general()
#================online strategy=================#
apt.online(output="phi")

#================offline strategy=================#
# # algorithm: DE
# DE_para = {"p_num":10, "deltaphi0":[], "max_episode":1000, "c":1.0, 
#            "cr":0.5, "seed":1234}
# apt.offline(method="DE", **DE_para)

# # algorithm: PSO
# PSO_para = {"p_num":10, "deltaphi0":[], "max_episode":[1000,100], 
#             "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
# apt.offline(method="PSO", **PSO_para)
