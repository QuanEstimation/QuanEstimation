import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class PSO(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], control_option=True, ctrl_bound=[-np.inf, np.inf], W=[], particle_num=10, ini_particle=[], \
                 max_episodes=[1000, 100], c0=1.0, c1=2.0, c2=2.0, v0=0.1, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option, ctrl_bound, W)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: number of particles.
           --type: int
        
        ini_particle:
           --description: initial particles.
           --type: array

        max_episodes:
            --description: max number of training episodes.
            --type: int
        
        c0:
            --description: damping factor that assists convergence.
            --type: float

        c1:
            --description: exploitation weight that attract the particle to its best previous position.
            --type: float
        
        c2:
            --description: exploitation weight that attract the particle to the best position in the neighborhood.
            --type: float

        v0:
            --description: the amplitude of the initial velocity.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """
        if ini_particle == []: 
            ini_particle = [ctrl_initial]

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episodes = max_episodes
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.v0 = v0
        self.seed = seed
    
    def QFIM(self, save_file=False):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients  
                     that maximize the QFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients for each episode but overwrite in the nest episode and all the QFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        Main.QuanEstimation.PSO_QFIM(pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)

    def CFIM(self, Measurement, save_file):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients  
                     that maximize the CFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the nest episode and all the CFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)