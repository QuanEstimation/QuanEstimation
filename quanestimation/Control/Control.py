import numpy as np
import warnings
import math

class ControlSystem:
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], control_option=True):
        
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
        
        """   
        
        self.tspan = tspan
        self.rho_initial = rho_initial.view(dtype=np.complex128)
        self.freeHamiltonian = H0.view(dtype=np.complex128)
        self.control_Hamiltonian = [x.view(dtype=np.complex128) for x in Hc]
        self.Hamiltonian_derivative = [x.view(dtype=np.complex128) for x in dH]
        self.control_coefficients = ctrl_initial
        self.Liouville_operator = [x.view(dtype=np.complex128) for x in Liouville_operator]
        self.gamma = gamma
        self.control_option = control_option
        self.tnum = len(tspan)

        if len(self.gamma) != len(self.Liouville_operator):
            raise TypeError('The length of decay rates and Liouville operators should be the same') 
        
        if type(self.Hamiltonian_derivative) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!')    
        

        ctrl_length = len(self.control_coefficients)
        ctrlnum = len(self.control_Hamiltonian)
        if ctrlnum < ctrl_length:
            raise TypeError('There are %d control Hamiltonians but %d coefficients sequences: \
                            too many coefficients sequences'%(ctrlnum,ctrl_length))
        elif ctrlnum > ctrl_length:
            warnings.warn('Not enough coefficients sequences: there are %d control Hamiltonians \
                           but %d coefficients sequences. The rest of the control sequences are\
                           set to be 0.'%(ctrlnum,ctrl_length), DeprecationWarning)
        
            number = math.ceil(self.tnum/ctrl_length)
            if self.tnum % ctrl_length != 0:
                self.tnum = number*ctrl_length
                self.tspan = np.linspace(self.tspan[0], self.tspan[-1], self.tnum)