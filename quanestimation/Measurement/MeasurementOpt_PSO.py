import numpy as np
from julia import Main
import quanestimation.Measurement.Measurement_struct as Measurement_struct

class PSO_Meas(Measurement_struct.MeasurementSystem):
    def __init__(self, tspan, rho0, H0, dH=[], decay=[], W=[], ini_measurement=[],  \
                particle_num=10, ini_particle=[], max_episode=[1000, 100], c0=1.0, c1=2.0, c2=2.0, seed=1234):

        Measurement_struct.MeasurementSystem.__init__(self, tspan, rho0, H0, dH, decay, W, ini_measurement, seed, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        ini_particle:
           --description: initial particles.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int or array
        
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
            ini_particle = [np.array(self.Measurement)]

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        
    def CFIM(self, save_file=False):
        """
        Description: use particle swarm optimization algorithm to update the measurements that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the measurements for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the measurements for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        pso = Main.QuanEstimation.MeasurementOpt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                 self.decay_opt, self.gamma, self.Measurement, self.W, self.accuracy)
        Main.QuanEstimation.PSO_CFIM(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, \
                                     self.seed, save_file)
        self.load_save()
        