import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class PSO(Control.ControlSystem):
    def __init__(self, tspan, rho0, H0, Hc=[], dH=[], ctrl_0=[], decay=[], ctrl_bound=[], W=[], \
                 particle_num=10, ini_particle=[], max_episode=[1000, 100], c0=1.0, c1=2.0, c2=2.0, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho0, H0, Hc, dH, ctrl_0, decay, ctrl_bound, W, accuracy=1e-8)
        
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

        max_episode:
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
        
        seed:
            --description: random seed.
            --type: int
        
        """
        if ini_particle == []: 
            ini_particle = [np.array(self.control_coefficients)]

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
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
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan, \
                        self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W, self.accuracy)
        Main.QuanEstimation.PSO_QFIM(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, \
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
        pso = Main.QuanEstimation.PSO(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan, \
                        self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W, self.accuracy)
        Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, \
                                         self.seed, save_file)