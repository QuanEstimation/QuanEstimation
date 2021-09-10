import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class PSO(Control.ControlSystem):
    def __init__(self, particle_num, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], episode=400, seed=100, c0=1.0, c1=2.0, c2=2.0, v0=0.01,control_option=True):
        
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
        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option)
        self.particle_num = particle_num
        self.episode = episode
        self.ctrlnum = len(Hc)
        self.ctrl_dim = len(ctrl_initial[0])
        self.ctrl_dim_total = self.ctrlnum*self.ctrl_dim
        self.X = np.zeros((particle_num,self.ctrl_dim_total))     #the position and velocity of all the particle
        self.V = np.zeros((particle_num,self.ctrl_dim_total))
        self.pbest = np.zeros((particle_num,self.ctrl_dim_total))  #the personal best position of a particle 
        self.gbest = np.zeros(particle_num)                #the global best position of a particle 
        self.p_fit = np.zeros(particle_num)            #target function of a particle
        self.fit = 0.0
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.v0 = v0
        self.seed = seed
        self.rho = None
        self.rho_derivative = None
        self.F = None
    
    def QFIM(self, save_file=False):
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients)
        Main.QuanEstimation.PSO_QFIM(pso, self.episode, particle_num=self.particle_num, c0=self.c0, c1=self.c1, c2=self.c2, \
                                     v0=self.v0, sd=self.seed)
         
    
    def swarm_origin(self):
        np.random.seed(self.seed)
        for pi in range(self.particle_num):
            self.X[pi] = np.random.random(self.ctrl_dim_total)
            self.V[pi] = np.random.random(self.ctrl_dim_total)
            
    def iteration(self):
        for pi in range(self.particle_num):
            #==============================================
            #evolve the dynamics with the controls
            self.control_coefficients = self.X[pi].reshape(self.ctrlnum,self.ctrl_dim)
            self.data_generation()
            self.propagator_save = []
            rho_final = mat_vec_convert(self.rho[self.tnum-1])
            drho_final = mat_vec_convert(self.rho_derivative[self.tnum-1][0])
            #calculate the objective function
            self.F = QFIM(rho_final,[drho_final])
            #==============================================
                
            if self.F > self.p_fit[pi]:
                self.p_fit[pi] = self.F
                self.pbest[pi] = self.X[pi]
            
        for pj in range(self.particle_num):
            if self.p_fit[pj] > self.fit:
                self.fit = self.p_fit[pj]
                self.gbest = self.X[pj]
                    
        np.random.random(self.seed)
        for pk in range(self.particle_num):
            self.V[pk] = self.c0*self.V[pk] + self.c1*np.random.random()*(self.pbest[pk] - self.X[pk]) + \
                        self.c2*np.random.random()*(self.gbest - self.X[pk])  
            self.X[pk] = self.X[pk] + self.V[pk]
                  