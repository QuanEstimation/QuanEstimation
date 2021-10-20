import numpy as np
from time import time
from quanestimation import *
from qutip import *

N = 6
#initial state
psi0 = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type='bra')
psi0 = psi0.full()[0]
#Hamiltonian
Lambda = 0.5
h = 0.1
Jx, Jy, Jz = jmat(0.5*N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -(np.dot(Jx, Jx)+Lambda*np.dot(Jy, Jy))/N-h*Jz
dH0 = [-np.dot(Jy, Jy)/N]
#dissipation
L_opt = [Jz]
gamma = [0.1]

T = 1.0
tnum = int(500*T)
tspan = np.linspace(0.0, T, tnum)
#initial psi0 for DE, PSO and NM
ini_state = [psi0]

# #AD algorithm
# AD = StateOpt_AD(tspan, psi0, H0, dH=dH0, Liouville_operator=L_opt, gamma=gamma, ctrl_bound=0.2, lr=0.01, epsilon=1e-8, max_episodes=300, Adam=False)
# AD.QFIM(save_file=True)

# #DE algorithm
diffevo = StateOpt_DE(tspan, psi0, H0, dH=dH0, Liouville_operator=L_opt, gamma=gamma, ctrl_bound=0.2, popsize=10, ini_population=ini_state,\
                      c=0.5, c0=0.1, c1=0.6, seed=1234, max_episodes=1000)
diffevo.QFIM(save_file=True)

# #PSO algorithm
# pso = StateOpt_PSO(tspan, psi0, H0, dH=dH0, Liouville_operator=L_opt, gamma=gamma, ctrl_bound=0.2, particle_num=10, ini_particle=ini_state, \
#                    max_episodes=[1000, 100], seed=1234, c0=1.0, c1=2.0, c2=2.0, v0=0.01)
# pso.QFIM(save_file=True)

# #Nelder Mead algorithm
# NM = NelderMead(tspan, psi0, H0, dH=dH0, Liouville_operator=L_opt, gamma=gamma, ctrl_bound=0.2, state_num=10, ini_state=ini_state,\
#                 coeff_r=1.0, coeff_e=2.0, coeff_c=0.5, coeff_s=0.5, seed=200, max_episodes=1000, epsilon=1e-3)
# NM.QFIM(save_file=True)
