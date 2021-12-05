import numpy as np
import warnings
import math
import os
import quanestimation.Control as ctrl
class ControlSystem:
    def __init__(self, tspan, rho0, H0, Hc, dH, ctrl_0, decay, ctrl_bound, W, accuracy):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        rho0: 
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
           
        ctrl_0: 
           --description: control coefficients.
           --type: list (of array)
           
        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0] represent a list of decay operators and
                          decay[1] represent the corresponding decay rates.
           --type: list 

        ctrl_bound:   
           --description: lower and upper bound of the control coefficients.
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
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0] 
        
        if Hc == []:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!') 

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]
        
        if ctrl_0 == []:
            if ctrl_bound == []:
                ctrl_0 = [2*np.random.random(len(self.tspan)-1)-np.ones(len(self.tspan)-1) for i in range(len(self.control_Hamiltonian))]
            else:
                a = ctrl_bound[0]
                b = ctrl_bound[1]
                ctrl_0 = [(b-a)*np.random.random(len(self.tspan)-1)+a*np.ones(len(self.tspan)-1) for i in range(len(self.control_Hamiltonian))]
        self.control_coefficients = ctrl_0
        
        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]
        if ctrl_bound == []:
            ctrl_bound = [-np.inf, np.inf]
        self.ctrl_bound = ctrl_bound
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.accuracy = accuracy
        
        if os.path.exists('controls.csv'):
            data = np.genfromtxt('controls.csv')[-len(self.control_Hamiltonian):]
            self.control_coefficients = [data[i] for i in range(len(data))]
            
        ctrl_length = len(self.control_coefficients)
        ctrlnum = len(self.control_Hamiltonian)
        if ctrlnum < ctrl_length:
            raise TypeError('There are %d control Hamiltonians but %d coefficients sequences: \
                                too many coefficients sequences'%(ctrlnum,ctrl_length))
        elif ctrlnum > ctrl_length:
            warnings.warn('Not enough coefficients sequences: there are %d control Hamiltonians \
                            but %d coefficients sequences. The rest of the control sequences are\
                            set to be 0.'%(ctrlnum,ctrl_length), DeprecationWarning)
        
        number = math.ceil((len(self.tspan)-1)/len(self.control_coefficients[0]))
        if len(self.tspan)-1 % len(self.control_coefficients[0]) != 0:
            tnum = number*len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum+1)

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
