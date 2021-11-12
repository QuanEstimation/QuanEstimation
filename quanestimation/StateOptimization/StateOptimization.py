import numpy as np
import os
import quanestimation.StateOptimization as stateoptimize
class StateOptSystem:
    def __init__(self, tspan, psi_initial, H0, dH, Liouville_operator, gamma, W):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        psi_initial:
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
           
        Liouville operator:
           --description: Liouville operator.
           --type: list (of matrix)    
           
        gamma:
           --description: decay rates.
           --type: list (of float number)

        W:
            --description: weight matrix.
            --type: matrix
        
        """   
        
        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!')    
        
        if len(gamma) != len(Liouville_operator):
            raise TypeError('The length of decay rates and the length of Liouville operator should be the same!')
        
        if dH == []:
            dH = [np.zeros((len(H0), len(H0)))]

        if Liouville_operator == []:
            Liouville_operator = [np.zeros((len(H0), len(H0)))]
        
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

        self.tspan = tspan
        self.psi_initial = np.array(psi_initial,dtype=np.complex128)
        self.freeHamiltonian = np.array(H0,dtype=np.complex128)
        self.Hamiltonian_derivative = [np.array(x,dtype=np.complex128) for x in dH]
        self.Liouville_operator = [np.array(x, dtype=np.complex128) for x in Liouville_operator]
        self.gamma = gamma
        
        if os.path.exists('states.csv'):
            self.psi_initial = np.genfromtxt('states.csv', dtype=np.complex128)

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