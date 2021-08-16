import numpy as np
from quanestimation.AsymptoticBound.CramerRao import QFIM
from quanestimation.Dynamics.dynamics import Lindblad
from quanestimation.Common.common import mat_vec_convert

class DiffEvo(Lindblad):
    def __init__(self, particle_num, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, gamma, control_option=True, c0=0.1, c1=0.6, seed=100):
        
        Lindblad.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                        gamma, control_option)
        """
        --------
        inputs
        --------
        particle_num:
           --description: number of particles.
           --type: float
        
        particle_num:
           --description: number of particles.
           --type: float
        
        """
        self.particle_num = particle_num
        self.ctrlnum = len(Hc)
        self.ctrl_dim = len(ctrl_initial[0])
        self.ctrl_dim_total = self.ctrlnum*self.ctrl_dim
        self.control = np.zeros((particle_num,self.ctrl_dim_total))
        
        self.p_fit = np.zeros(particle_num)
    
        self.c0 = c0
        self.c1 = c1
        self.seed = seed
        self.tnum = len(tspan)
        self.rho = None
        self.rho_derivative = None
        self.F = None
        
    def initialize(self):
        np.random.seed(self.seed)
        for pi in range(self.particle_num):
            self.control[pi] = np.random.random(self.ctrl_dim_total)
            
            #======calculate the target function of the initialized population======
            self.control_coefficients = self.control[pi].reshape(self.ctrlnum,self.ctrl_dim)
            self.data_generation()
            self.propagator_save = []
            rho_final = mat_vec_convert(self.rho[self.tnum-1])
            drho_final = mat_vec_convert(self.rho_derivative[self.tnum-1][0])
            self.p_fit[pi] = QFIM(rho_final,[drho_final])
            #==============================================
            
    def mutation(self):
        #?????????
        for pi in range(self.particle_num):
            r1,r2,r3 = 0,0,0
            while r1 == pi or r2 == pi or r3 == pi or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = np.random.randint(0, self.particle_num - 1)
                r2 = np.random.randint(0, self.particle_num - 1)
                r3 = np.random.randint(0, self.particle_num - 1)
                
            mut = self.control[r1] + self.F*(self.control[r2]-self.control[r3])
        return mut
    
    def crossover(self):
        mut = self.mutation()
        f_mean = np.mean(self.p_fit)
        
        for pi in range(self.particle_num):
            #==============Adaptive adjustment Cr=======================
            if self.p_fit[pi] > f_mean:
                Cr = self.c0+(self.c1-self.c0)*(self.p_fit[pi]-min(self.p_fit))/(max(self.p_fit)-min(self.p_fit))
            else:
                Cr = self.c0
                
            set_cross = np.zeros(self.ctrl_dim_total)
            rand_ind = np.random.randint(0, self.ctrl_dim_total-1)
            for ck in range(self.ctrl_dim_total):
                rand_r = np.random.random() 
                if rand_r <= Cr or rand_ind == pi:   
                    set_cross[ck] = mut[ck]
                else:
                    set_cross[ck] = self.control[pi][ck]
             
            #============calculate fitness===============  
            self.control_coefficients = set_cross.reshape(self.ctrlnum,self.ctrl_dim)
            self.data_generation()
            self.propagator_save = []
            rho_final = mat_vec_convert(self.rho[self.tnum-1])
            drho_final = mat_vec_convert(self.rho_derivative[self.tnum-1][0])
            p_cross = QFIM(rho_final,[drho_final])
            #==============================================
            
            if p_cross < self.p_fit[pi]:
                self.control[pi] = set_cross
                self.p_fit[pi] = p_cross
                  