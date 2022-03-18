import numpy as np
from quanestimation import *

N = 8
a = annihilation(N + 1)

psi = np.zeros((N + 1) ** 2).reshape(-1, 1)
for k in range(N + 1):
    psi += np.sin((k + 1) * np.pi / (N + 2)) * np.kron(
        basis(N + 1, k), basis(N + 1, N - k)
    )
psi = np.sqrt(2 / (2 + N)) * psi
rho0 = np.dot(psi, psi.conj().T)

x = np.linspace(-np.pi, np.pi, 100)
p = (1.0 / (x[-1] - x[0])) * np.ones(len(x))
apt = adaptMZI(x, p, rho0)
apt.general()
apt.online(output="phi")
