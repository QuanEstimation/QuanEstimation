import numpy as np
from julia import Main
from scipy.sparse import csr_matrix
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.Common.common import brgd

class adaptMZI():
    def __init__(self, x, p, rho0):
        
        self.x = x
        self.p = p
        self.rho0 = rho0
        self.N = int(np.sqrt(len(rho0))) -1 
        #### annihilation operator ####
        n = self.N + 1
        data = np.sqrt(np.arange(1, n, dtype=complex))
        indices = np.arange(1, n)
        indptr = np.arange(n+1)
        indptr[-1] = n-1
        self.a = csr_matrix((data, indices, indptr), shape=(n, n)).todense()
        
    def general(self):
        self.MZI_type = "general"
          
    def online(self, output="phi"):
        phi = Main.QuanEstimation.adaptMZI_online(self.x, self.p, self.rho0, self.a, output)
    
    def offline(self, method="DE", popsize=10, particle_num=10, DeltaPhi0=[], c=1.0, \
            cr=0.5, c0=1.0, c1=2.0, c2=2.0, seed=1234, max_episode=1000, eps=1e-8):
        comb_tp = brgd(self.N)
        comb = [np.array([int(list(comb_tp[i])[j]) for j in range(self.N)]) for i in range(2**self.N)]
        if method == "DE":
            Main.QuanEstimation.DE_DeltaPhiOpt(self.x, self.p, self.rho0, self.a, comb, popsize, \
                                                              DeltaPhi0, c, cr, seed, max_episode, eps)
        elif method == "PSO":
            Main.QuanEstimation.PSO_DeltaPhiOpt(self.x, self.p, self.rho0, self.a, comb, \
                                              particle_num, DeltaPhi0, c0, c1, c2, seed, max_episode, eps)
        else:
            raise ValueError("{!r} is not a valid value for method, supported values are 'DE' and 'PSO'.".format(method))
            
    
        