import numpy as np
import os
import quanestimation.StateOpt as stateoptimize

class StateSystem:
    def __init__(self, tspan, psi0, H0, dH, decay, W, accuracy):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        psi0:
            --description: initial guess of states (kets).
            --type: array
        
        H0: 
           --description: free Hamiltonian.
           --type: matrix (a list of matrix)
        
        dH: 
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)
           
        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0][0] represent the first decay operator and
                          decay[0][1] represent the corresponding decay rate.
           --type: list 

        ctrl_bound:   
           --description: lower and upper bounds of the control coefficients.
                          ctrl_bound[0] represent the lower bound of the control coefficients and
                          ctrl_bound[1] represent the upper bound of the control coefficients.
           --type: list 

        W:
            --description: weight matrix.
            --type: matrix

        accuracy:
            --description: calculation accuracy.
            --type: float
        """   
        self.tspan = tspan

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
            self.dim = len(self.freeHamiltonian)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
            self.dim = len(self.freeHamiltonian[0])

        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!') 
            
        if dH == []:
            dH = [np.zeros((len(self.psi0), len(self.psi0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]    
        
        if decay == []:
            decay_opt = [np.zeros((len(self.psi0), len(self.psi0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        self.accuracy = accuracy

        if psi0 == []:
            np.random.seed(seed)
            for i in range(self.dim):
                r_ini = 2*np.random.random(self.dim)-np.ones(self.dim)
                r = r_ini/np.linalg.norm(r_ini)
                phi = 2*np.pi*np.random.random(self.dim)
                psi0 = [r[i]*np.exp(1.0j*phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
        else:
            self.psi0 = np.array(psi0[0],dtype=np.complex128)

        if os.path.exists('states.csv'):
            self.psi0 = np.genfromtxt('states.csv', dtype=np.complex128)

    def load_save(self):
        file_load = open('states.csv', 'r')
        file_load = ''.join([i for i in file_load]).replace("im", "j")
        file_load = ''.join([i for i in file_load]).replace(" ", "")
        file_save = open("states.csv","w")
        file_save.writelines(file_load)
        file_save.close()

def StateOpt(*args, method = 'AD', **kwargs):

    if method == 'AD':
        return stateoptimize.AD_Sopt(*args, **kwargs)
    elif method == 'PSO':
        return stateoptimize.PSO_Sopt(*args, **kwargs)
    elif method == 'DE':
        return stateoptimize.DE_Sopt(*args, **kwargs)
    elif method == 'DDPG':
        return stateoptimize.DDPG_Sopt(*args, **kwargs)
    elif method == 'NM':
        return stateoptimize.NM_Sopt(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE', 'NM', 'DDPG'.".format(method))
        