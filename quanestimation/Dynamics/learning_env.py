import numpy as np
from scipy import linalg as scylin
import warnings
import julia
#julia.install()
# from julia import Main
from qutip import *

from quanestimation.AsymptoticBound.CramerRao import QFIM
from quanestimation.Common.common import dRHO

# Main.include('./'+'Common'+'/'+'Liouville.jl')

class env():
    def __init__(self, rho0, tspan, ctrl_length, H0, Hc=[], dH=[], Liouville_operator=[], gamma=[]):

        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
           
        ctrl_length: 
           --description: length of each control coefficient.
           --type: int   
           
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
           
        Liouville operator:
           --description: Liouville operator in Lindblad master equation.
           --type: list (of matrix) 
           
        gamma:
           --description: decay rates.
           --type: list (of float number)
           
        """
        self.gamma = gamma
        self.H0 = H0
        self.Hc = Hc
        self.dH = dH
        self.Liouville_operator = Liouville_operator
        self.rho0 = rho0
        
        self.para_num = len(dH)
        self.ctrl_num = len(Hc)
        self.Liou_num = len(Liouville_operator)
        self.H0_Liou = Main.Liouville.liouville_commu(H0)
        self.dim = len(H0)
        
        self.tspan = tspan
        self.tnum = len(tspan)
        self.dt = tspan[0]-tspan[1]
        self.ctrl_length = ctrl_length

        if type(dH) != list:
            raise TypeError('Please make sure dH is list!')
        else:
            self.dH_Liou = []
            for para_i in range(0,self.para_num):
                dH_temp = Main.Liouville.liouville_commu(dH[para_i])
                self.dH_Liou.append(dH_temp)
        
        if len(gamma) != self.Liou_num:
            raise TypeError('Please make sure to input the same number of decay rates and Liouville operator')
        else:
            self.Liou_part = []
            for ki in range(0,self.Liou_num):
                L_temp = Main.Liouville.liouville_dissip(Liouville_operator[ki])
                self.Liou_part.append(gamma[ki]*L_temp)
                
        for ci in range(ctrl_length):
            self.ctrl_interval = int(self.tnum/ctrl_length)
            if self.tnum%ctrl_length != 0:
                warnings.warn('the coefficients sequences and time number are not multiple')
        
        self.Hc_Liou = []
        for hi in range(0,self.ctrl_num):
            Hc_temp = Main.Liouville.liouville_commu(Hc[hi])
            self.Hc_Liou.append(Hc_temp)

        
    def input_state(self,rho):
        "rho: matrix (np.array)"
        "state: 2*dim**2 dimensional vector"
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

    def F0(self, state, dstate):
        Liouv_tot = -1.j*self.H0_Liou+self.Liou_part[0]
        rho_pre = self.state_to_Liouville(state)  
        rho = np.dot(scylin.expm(self.dt*Liouv_tot), rho_pre)
        
        A = (-1j*np.dot(self.dH_Liou,rho)).reshape(-1,1)
        dstate = dRHO(dstate, Liouv_tot, A, self.dt)
        rho = rho.reshape(self.dim, self.dim)
        f = QFIM(rho, [dstate.reshape(self.dim, self.dim)])
        return f

    
    def step(self, action, state, dstate):
        "state_in: 2*dim**2 dimensional vector"
        
        Hc = np.zeros((self.dim**2,self.dim**2),dtype=np.complex128)
        for hi in range(0, self.ctrl_num):
            Hc = Hc + float(action[hi])*self.Hc_Liou[hi]
        Liouv_tot = -1.j*(self.H0_Liou+Hc)+self.Liou_part[0]
        for i in range(self.ctrl_interval):
            rho_pre = self.state_to_Liouville(state)  
            rho = np.dot(scylin.expm(self.dt*Liouv_tot), rho_pre)
        
            A = (-1j*np.dot(self.dH_Liou,rho)).reshape(-1,1)
            dstate = dRHO(dstate, Liouv_tot, A, self.dt)
            rho = rho.reshape(self.dim, self.dim)
            
            state = self.input_state(rho)
        
        f0 = self.F0(state, dstate)  
        F_q = self.QFIM(rho, [dstate.reshape(self.dim, self.dim)])
        reward = F_q
        return state, dstate, reward