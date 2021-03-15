import numpy as np
from qutip import *
import time
import pyximport;pyximport.install()

from Common.common import Liouville_commu, Liouville_dissip

N = 6
M = 3
wc = 110.
wz = 100.
g = 1.0
nu = 200.
zeta = 3.113
t = 10.0

a = tensor(destroy(M), qeye(N+1)).full()
a_dag = tensor(destroy(M), qeye(N+1)).dag().full()
Jx = tensor(qeye(M), jmat(0.5*N, 'x')).full()
Jy = tensor(qeye(M), jmat(0.5*N, 'y')).full()
Jz = tensor(qeye(M), jmat(0.5*N, 'z')).full()
H0 = wc*np.dot(a_dag, a) + wz*Jz + g*np.dot(Jx, (a_dag + a))
Ht = zeta*nu*np.cos(nu*t)*Jz

def main():
    start = time.time()
    result = Liouville_dissip(a)
    #result = Liouville_commu(H0+Ht)
    duration = time.time()-start
    print(result,duration)

if __name__=='__main__':
    main()
