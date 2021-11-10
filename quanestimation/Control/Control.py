import numpy as np
import warnings
import math
import os
import quanestimation.Control as ctrl
class ControlSystem:
    def __init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                 gamma, control_option, ctrl_bound, W):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        rho_initial: 
           --description: initial state (density matrix).
           --type: matrix
        
        H0: 
           --description: free Hamiltonian.
           --type: matrix
           
        Hc: 
           --description: control Hamiltonian.
           --type: list (of matrix)
        
        dH: 
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)
           
        ctrl_initial: 
           --description: control coefficients.
           --type: list (of array)
           
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

        if Hc == []:
            Hc = [np.zeros((len(H0), len(H0)))]

        if ctrl_initial == []:
            ctrl_initial = [np.zeros(len(tspan))]
        
        if dH == []:
            dH = [np.zeros((len(H0), len(H0)))]

        if Liouville_operator == []:
            Liouville_operator = [np.zeros((len(H0), len(H0)))]

        if gamma == []:
            gamma = [0.0]
        
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]

        self.tspan = tspan
        self.rho_initial = np.array(rho_initial, dtype=np.complex128)
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]
        self.control_coefficients = ctrl_initial
        self.Liouville_operator = [np.array(x, dtype=np.complex128) for x in Liouville_operator]
        self.gamma = gamma
        self.control_option = control_option
        self.ctrl_bound = ctrl_bound
        
        ctrl_length = len(self.control_coefficients)
        ctrlnum = len(self.control_Hamiltonian)
        if ctrlnum < ctrl_length:
            raise TypeError('There are %d control Hamiltonians but %d coefficients sequences: \
                                too many coefficients sequences'%(ctrlnum,ctrl_length))
        elif ctrlnum > ctrl_length:
            warnings.warn('Not enough coefficients sequences: there are %d control Hamiltonians \
                            but %d coefficients sequences. The rest of the control sequences are\
                            set to be 0.'%(ctrlnum,ctrl_length), DeprecationWarning)
        
        number = math.ceil(len(self.tspan)/len(self.control_coefficients[0]))
        if len(self.tspan) % len(self.control_coefficients[0]) != 0:
            self.tnum = number*len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], self.tnum)

        if os.path.exists('controls.csv'):
            data = np.genfromtxt('controls.csv')
            self.control_coefficients = [data[i] for i in range(len(data))]

def ControlOpt(*args, method = 'auto-GRAPE', **kwargs):

    if method == 'auto-GRAPE':
        return ctrl.GRAPE(*args, **kwargs, auto=True)
    elif method == 'GRAPE':
        return ctrl.GRAPE(*args, **kwargs, auto=False)
    elif method == 'PSO':
        return ctrl.PSO(*args, **kwargs)
    elif method == 'DE':
        return ctrl.DiffEvo(*args, **kwargs)
    elif method == 'DDPG':
        return ctrl.DDPG(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'auto-GRAPE', 'GRAPE', 'PSO', 'DE', 'DDPG'.".format(method))