import numpy as np
from qutip import *

class env(object):
    def __init__(self, ctrl_interval, tnum_iteration, eta):
        super(env, self).__init__()
        self.ctrl_interval = ctrl_interval
        self.tnum_iteration = tnum_iteration
        self.eta = eta
        self.sy = sigmay()
        self.sz = sigmaz()
        
        #initial density matrix
        basis_p = (basis(2,1)+basis(2,0))/(2**0.5)
        self.rho0 = basis_p*basis_p.dag()
        
        self.dim = self.rho0.shape[0]
        self.n_actions = 2
        self.n_states = 2*self.dim**2
        
    def input_state(self,rho):
        "rho: matrix (np.array)"
        state_tp = np.append(np.reshape(np.array(rho.real), (self.dim**2, 1)), \
                      np.reshape(np.array(rho.imag), (self.dim**2, 1)), axis=1)
        state = state_tp.flatten()
        return state

    def state_to_Liouville(self,state):
        "state: 2*dim**2 dimensional vector"
        "rho_L: dim**2 dimensional vector"
        rho_tp = state.reshape([self.dim**2, 2])
        rho_L = rho_tp[:, 0] + 1j*rho_tp[:, 1]
        return rho_L

    def step(self,state_in, V, t):
        "state_in: 2*dim**2 dimensional vector"
 
        tspan = np.linspace(t, t+self.ctrl_interval, self.tnum_iteration)
        H = V[0]*self.sy + V[1]*self.sz
        rho_pre = self.state_to_Liouville(state_in).reshape([self.dim, self.dim])
    
        rho = mesolve(H, Qobj(rho_pre), tspan , [], [])
    
        #reward r_{j+1}:
        Fidelity = fidelity(self.rho0, rho.states[-1])
        reward = 10*(1.0 - self.eta*Fidelity)
 
        state = self.input_state(rho.states[-1].full())
        return state, reward, Fidelity