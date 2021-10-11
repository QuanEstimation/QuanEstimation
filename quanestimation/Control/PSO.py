import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class PSO(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], control_option=True, ctrl_bound=10.0, W=[], particle_num=10, max_episodes=400, \
                 seed=100, c0=1.0, c1=2.0, c2=2.0, v0=0.01):
        
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
        self.max_episodes = max_episodes
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
        self.ctrl_bound = ctrl_bound
        self.rho = None
        self.rho_derivative = None
        self.F = None
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W
    
    def QFIM(self, save_file=False):
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        if len(self.Hamiltonian_derivative) == 1:
            Main.QuanEstimation.PSO_QFI(pso, self.max_episodes, self.particle_num, self.c0, self.c1, self.c2, self.v0, \
                                        self.seed, save_file)
        else:
            Main.QuanEstimation.PSO_QFIM(pso, self.max_episodes, self.particle_num, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
         