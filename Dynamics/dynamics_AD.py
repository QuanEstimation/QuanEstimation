# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:19:07 2020

@author: JL MZ
"""
import numpy as np
import warnings
import math
from scipy import linalg as scylin
from time import time

from julia.api import Julia
jl = Julia(compiled_modules=False)
import julia
#julia.install()
from julia import Main
# import juliacall
# from juliacall import Main

Main.include("./Dynamics/dynamics.jl")


class Lindblad_AD:
    """
    General dynamics of density matrices in the form of time local Lindblad master equation.
    {\partial_t \rho} = -i[H, \rho] + \sum_n {\gamma_n} {Ln.rho.Ln^{\dagger}
                 -0.5(rho.Ln^{\dagger}.Ln+Ln^{\dagger}.Ln.rho)}.
    """

    def __init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator=[], gamma=[], control_option=True):
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
           --description: Liouville operator in Lindblad master equation.
           --type: list (of matrix)    
           
        gamma:
           --description: decay rates.
           --type: list (of float number)
           
        control_option:   
           --description: if True, add controls to physical system.
           --type: bool
        """
        
        self.freeHamiltonian = H0
        self.Liouville_operator = Liouville_operator
        self.gamma = gamma
        self.rho_initial = rho_initial
        self.tspan = tspan
        self.dt = self.tspan[1]-self.tspan[0]
        self.T = self.tspan[-1]
        self.tnum = len(self.tspan)
        self.Hamiltonian_derivative = dH
        self.control_Hamiltonian = Hc
        self.control_coefficients = ctrl_initial
        self.control_option = control_option

        self.dim = len(self.freeHamiltonian)
        self.Liouvillenumber = len(self.Liouville_operator)
        
        self.ctrlnum = len(self.control_Hamiltonian)
        self.ctrlnum_total = self.ctrlnum
        self.control_coeff_total = self.control_coefficients
        self.rho = None
        self.rho_derivative = None
        self.propagator_save = None
        self.environment_assisted_order = None
        self.environmentstate = False
        
        if self.control_option == True:
            ctrl_length = len(self.control_coefficients)
            if self.ctrlnum < ctrl_length:
                raise TypeError('there are too many coefficients sequences: exit the program')
            else:
                if self.ctrlnum > ctrl_length:
                    warnings.warn('not enough coefficients sequences: there are %d control Hamiltonians \
                              but %d coefficients sequences. We set the other control coefficients \
                              to 0.'%(self.ctrlnum,ctrl_length), DeprecationWarning)
                    num = int(self.ctrlnum-ctrl_length)
                    for ci in range(num):
                        ctrl_ci = np.zeros(self.tnum)
                        self.control_coefficients.append(ctrl_ci)
                        
                for i in range(len(ctrl_initial)):
                    number = int(self.tnum/len(ctrl_initial[i]))
                    if self.tnum%len(ctrl_initial[i]) != 0:
                        warnings.warn('the coefficients sequences and time number are not multiple: \
                                   we will set the non-divisible part to 0') 
                        ctrl_initial[i] = ctrl_initial[i].repeat(number)
                        ctrl_initial[i] = np.concatenate((ctrl_initial[i], np.zeros(int(self.tnum\
                                               -len(ctrl_initial[i])))))
                    else:
                        ctrl_initial[i] = ctrl_initial[i].repeat(number)   
        else: pass

        
        if len(self.gamma) != self.Liouvillenumber:
            raise TypeError('Please make sure to input the same number of decay rates and Liouville operator')
        
        if type(self.Hamiltonian_derivative) != list:
            raise TypeError('Please make sure dH is list!')
        else:
            self.freeHamiltonian_derivative_Liou = []
            for para_i in range(0,len(self.Hamiltonian_derivative)):
                dH0_temp = Main.liouville_commu(self.Hamiltonian_derivative[para_i])
                self.freeHamiltonian_derivative_Liou.append(dH0_temp)
         #------------------------------------------------------------------------------------------       
        #Generation of the Liouville representation of control Hamiltonians: self.ctrlH_Liou.
        #------------------------------------------------------------------------------------------
        self.ctrlH_Liou = []
        for hi in range(0,self.ctrlnum):
            Htemp = Main.liouville_commu(self.control_Hamiltonian[hi])
            self.ctrlH_Liou.append(Htemp)

    def general_information(self):
        print('==================================')
        print('General information:')
        show_dimension = 'dimension of Hamiltonian: '+str(self.dim)
        print(show_dimension)
        show_Liou = 'number of Liouville operators: '+str(self.Liouvillenumber)
        print(show_Liou)
        show_num = 'number of time step: '+str(self.tnum)
        print(show_num)
        show_ctrl = 'number of controls: '+str(self.ctrlnum_total)
        print(show_ctrl)
        show_cswitch = 'Control switch is '+str(self.control)
        print(show_cswitch)
        print('==================================')


    def data_generation(self):
        """
        Description: This function will save all the propators during the evolution,
                     which may be memory consuming.
        ----------
        outputs
        ----------
        rho: 
           --description: parameterized density matrix.
           --type: list (of matrix)
           
        rho_derivative: 
           --description: derivatives of density matrix on all parameters to
                          be estimated.
           --type: list (of matrix)
           
        propagator_save: 
           --description: propagating superoperator.
           --type: list (of matrix)  
        """ 
        tnum = self.tnum
        dim = self.dim
        dH = self.Hamiltonian_derivative 
        para_len = len(dH)
        dt = self.dt
        
        rhovec = [[] for i in range(0,tnum)]
        drhovec = [[[] for k in range(0,para_len)] for i in range(0,tnum)]



        rhovec, drhovec = Main.evolution(self.freeHamiltonian, dH, self.rho_initial, self.Liouville_operator, self.gamma,self.control_Hamiltonian, self.control_coefficients, self.tspan)

        self.rho = rhovec
        self.rho_derivative = drhovec

    def environment_assisted_state(self,statement,Dissipation_order):
        '''
        If the dissipation coefficient can be manually manipulated, it can be updated via GRAPE.
        This function is used to clarify which dissipation parameter can be updated.
        Input: 1) statement: True: the dissipation parameter is involved in the GRAPE.
                       False: the dissipation parameter is not involved in the GRAPE.
             2) Dissipation_order: number list contains the number of dissipation parameter to be updated.
                            For example, [3] means the 3rd Liouville operator can be updated and
                            [3, 5] means the 3rd and 5th Liouville operators can be updated.
        '''
        if  statement == True:
            newnum = int(self.ctrlnum+len(Dissipation_order))
            Hk_Liou = [[] for i in range(0,newnum)] 
            for hi in range(0,self.ctrlnum):
                Hk_Liou[hi] = Main.liouville_commu(self.control_Hamiltonian[hi])
            for hi in range(0,len(Dissipation_order)):
                hj = int(self.ctrlnum+hi)
                hnum = Dissipation_order[hi]
                Hk_Liou[hj] = 1.j*Main.liouville_dissip(self.Liouville_operator[hnum])
                ga = self.gamma[hnum]
                ctrl_coeff = self.control_coeff_total
                ctrl_coeff.append(ga)
                self.control_coeff_total = ctrl_coeff
            self.ctrlnum_total = newnum
            self.ctrlH_Liou = Hk_Liou
            self.environment_assisted_order = Dissipation_order
            self.environmentstate = statement