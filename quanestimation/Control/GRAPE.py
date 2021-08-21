import numpy as np
import warnings
import math
# import julia
from julia import Main
# from julia import QuanEstimation

class GRAPE:
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
             gamma=[], W=[], max_epsides=200, epsilon=1e-4, lr=0.01, precision=1e-8):
        
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
           --description: Weight matrix.
           --type: matrix

        lr:
           --description: learning.
           --type: float number

        precision:
           --description: calculation precision.
           --type: float number
         
       epsilon:
           --description: stop condition.
           --type: float number
        
        """   
        
        self.tspan = tspan
        self.rho_initial = rho_initial
        self.freeHamiltonian = H0
        self.control_Hamiltonian = Hc
        self.Hamiltonian_derivative = dH
        self.control_coefficients = ctrl_initial
        self.Liouville_operator = Liouville_operator
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.precision = precision
        self.max_epsides = max_epsides
        
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

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

    def QFIM(self, auto=True, save_file=False):
        """
        Description: use GRAPE algorithm to update the ****.
        
        ---------
        Inputs
        ---------
        auto:
           --description: True: use autodifferential to calculate the gradient. 
                          False: calculate the gradient with analytical method.
           --type: bool 
           
        save_file:
           --description: True: save all the control coefficients and quantum fisher information 
                          for single parameter estimation and the value of target function 
                          for multiparameter estimation.
                          False: return quantum fisher information for single parameter 
                          estimation and the value of target function for multiparameter estimation.
           --type: bool 
           
        ----------
        Returns
        ----------
           --description: updated values of control coefficients and for single parameter estimation 
                          and the value of target function for multiparameter estimation.
           --type: txt file

        ----------
        Notice
        ----------
           1) maximize is always more accurate than the minimize in this code.
        
        """
        grape = Main.QuanEstimation.Gradient(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                    self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.W, self.lr, self.precision)
        if auto == True:
            Main.QuanEstimation.GRAPE_QFIM_auto(grape, self.epsilon, self.max_epsides, save_file)
        else:
            Main.QuanEstimation.GRAPE_QFIM_analy(grape, self.epsilon, self.max_epsides, save_file)
        
    def CFIM(self, Measurement, auto=True, save_file=False):
        """
        Description: ****.
                     
        ---------
        Inputs
        ---------
        M:
           --description: a set of POVM. It takes the form [M1, M2, ...].
           --type: list (of matrix)
           
        auto:
           --description: True: use autodifferential to calculate the gradient. 
                          False: calculate the gradient with analytical method.
           --type: bool 
           
        save_file:
           --description: True: save all the control coefficients and classical fisher information 
                          for single parameter estimation and the value of target function 
                          for multiparameter estimation.
                          False: return classical fisher information for single parameter 
                          estimation and the value of target function for multiparameter estimation.
           --type: bool  
        
        ----------
        Returns
        ----------
           --description: updated values of control coefficients and for single parameter estimation 
                          and the value of target function for multiparameter estimation.
           --type: txt file

        ----------
        Notice
        ----------
           1) maximize is always more accurate than the minimize in this code.
        
        """
        
        grape = Main.QuanEstimation.Gradient(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                    self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.W, self.lr)
        if auto == True:
            Main.QuanEstimation.GRAPE_QFIM_auto(Measurement, grape, self.epsilon, self.max_epsides, save_file)
        else:
            Main.QuanEstimation.GRAPE_QFIM_analy(Measurement, grape, self.epsilon, self.max_epsides, save_file)