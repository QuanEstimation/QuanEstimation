import numpy as np
from scipy import linalg as scylin

from AsymptoticBound.CramerRao import CramerRao
from Common.common import mat_vec_convert, Liouville_commu, Liouville_dissip, dRHO

class env(CramerRao):
    def __init__(self, dt, H0, Hc=[], dH0=[], Liouville_operator=[], gamma=[]):

        CramerRao.__init__(self)
        
        self.gamma = gamma
        self.dt = dt
        self.H0 = H0
        self.Hc = Hc
        self.dH0 = dH0
        self.Liouville_operator = Liouville_operator
        
        self.para_num = len(dH0)
        self.ctrl_num = len(Hc)
        self.Liou_num = len(Liouville_operator)
        self.dim = len(H0)

        self.H0_Liou = Liouville_commu(H0)

        if type(dH0) != list:
            raise TypeError('Please make sure dH0 is list!')
        else:
            self.dH0_Liou = []
            for para_i in range(0,self.para_num):
                dH0_temp = Liouville_commu(dH0[para_i])
                self.dH0_Liou.append(dH0_temp)
        
        if len(gamma) != self.Liou_num:
            raise TypeError('Please make sure to input the same number of decay rates and Liouville operator')
        else:
            self.Liou_part = []
            for ki in range(0,self.Liou_num):
                L_temp = Liouville_dissip(Liouville_operator[ki])
                self.Liou_part.append(gamma[ki]*L_temp)
                
        self.Hc_Liou = []
        for hi in range(0,self.ctrl_num):
            Hc_temp = Liouville_commu(Hc[hi])
            self.Hc_Liou.append(Hc_temp)

        
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

    def step(self, action, state_in, drho_in):
        "state_in: 2*dim**2 dimensional vector"
        
        Hc = np.zeros((self.dim**2,self.dim**2),dtype=np.complex128)
        for hi in range(0, self.ctrl_num):
            Hc = Hc + action[hi]*self.Hc_Liou[hi]
        Liouv_tot = -1.j*(self.H0_Liou+Hc)+self.Liou_part[0]
        
        rho_pre = self.state_to_Liouville(state_in)  
        rho = np.dot(scylin.expm(self.dt*Liouv_tot), rho_pre)
        
        A = (-1j*np.dot(self.dH0_Liou,rho)).reshape(-1,1)
        drho = dRHO(drho_in, Liouv_tot, A, self.dt)
        
        rho = rho.reshape(self.dim, self.dim)

        F_q = self.QFIM(rho, [drho.reshape(self.dim, self.dim)])
        
        reward = F_q
        
        state = self.input_state(rho)
        return state, drho, reward, F_q