import numpy as np
import os
import quanestimation.StateOptimization as stateoptimize
class StateOptSystem:
    def __init__(self, tspan, psi0, H0, dH, Decay, W):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        psi0:
            --description: initial state.
            --type: array
        
        H0: 
           --description: free Hamiltonian.
           --type: matrix
        
        dH: 
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)
           
        Decay:
           --description: decay operators and the corresponding decay rates.
                          Decay[0] represent a list of decay operators and
                          Decay[1] represent the corresponding decay rates.
           --type: list 

        ctrl_bound:   
           --description: lower and upper bound of the control coefficients.
                          ctrl_bound[0] represent the lower bound of the control coefficients and
                          ctrl_bound[1] represent the upper bound of the control coefficients.
           --type: list 

        W:
            --description: weight matrix.
            --type: matrix
        
        """   
        self.tspan = tspan
        self.psi0 = np.array(psi0,dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]

        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!') 
            
        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]    
        
        Decay_opt = Decay[0]
        if Decay_opt == []:
            Decay_opt = [np.zeros((len(self.freeHamiltonian), len(self.freeHamiltonian)))]
        self.Decay_opt = [np.array(x, dtype=np.complex128) for x in Decay_opt]

        gamma = Decay[1]
        if gamma == []:
            gamma = [0.0]
        self.gamma = gamma

        if len(self.gamma) != len(self.Decay_opt):
            raise TypeError('The length of decay rates and the length of Liouville operator should be the same!')
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
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
        return stateoptimize.StateOpt_AD(*args, **kwargs)
    elif method == 'PSO':
        return stateoptimize.StateOpt_PSO(*args, **kwargs)
    elif method == 'DE':
        return stateoptimize.StateOpt_DE(*args, **kwargs)
    elif method == 'NM':
        return stateoptimize.StateOpt_NM(*args, **kwargs)
    elif method == 'DDPG':
        return stateoptimize.StateOpt_DDPG(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE', 'DDPG'.".format(method))