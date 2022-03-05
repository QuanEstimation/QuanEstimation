import numpy as np
from quanestimation import *
from scipy.sparse import csr_matrix

def get_basis(dim, index):
    x = np.zeros(dim)
    x[index] = 1.0
    return x.reshape(dim,1)

N = 8
#### annihilation operator ####
n = N+1
data = np.sqrt(np.arange(1, n, dtype=complex))
indices = np.arange(1, n)
indptr = np.arange(n+1)
indptr[-1] = n-1
a = csr_matrix((data, indices, indptr), shape=(n, n)).todense()

psi = np.zeros((N+1)**2).reshape(-1,1)
for k in range(N+1):
    psi += np.sin((k+1)*np.pi/(N+2))*np.kron(get_basis(N+1, k), get_basis(N+1, N-k))
psi = np.sqrt(2/(2+N))*psi
rho0 = np.dot(psi, psi.conj().T)

x = np.linspace(-np.pi, np.pi, 100)
p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
apt = adaptMZI(x, p, rho0)
apt.general()
apt.online(output="phi")
