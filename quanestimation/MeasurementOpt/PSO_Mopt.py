import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement

class PSO_Mopt(Measurement.MeasurementSystem):
    def __init__(self, mtype, minput, tspan, rho0, H0, dH, Hc=[], ctrl=[], decay=[], W=[], particle_num=10, \
                measurement0=[], max_episode=[1000, 100], c0=1.0, c1=2.0, c2=2.0, seed=1234, load=False):

        Measurement.MeasurementSystem.__init__(self, mtype, minput, tspan, rho0, H0, dH, Hc, ctrl, decay, W, measurement0, seed, load, eps=1e-8)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        measurement0:
           --description: initial guesses of measurements.
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
        self.particle_num = particle_num
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

        if self.mtype == "projection":
            if measurement0 == []: 
                measurement0 = [np.array(self.M)]
            self.measurement0 = measurement0
        else: pass
        
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
        if self.mtype=="projection":
            pso = Main.QuanEstimation.projection_Mopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                    self.decay_opt, self.gamma, self.M, self.W, self.eps)
            Main.QuanEstimation.CFIM_PSO_Mopt(pso, self.max_episode, self.particle_num, self.measurement0, self.c0, self.c1, \
                                            self.c2, self.seed, save_file)
            self.load_save()

        elif self.mtype=="input":
            pso = Main.QuanEstimation.LinearComb_Mopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                    self.decay_opt, self.gamma, self.povm_basis, self.M_num, self.W, self.eps)
            Main.QuanEstimation.CFIM_PSO_Mopt(pso, self.max_episode, self.particle_num, self.c0, self.c1, self.c2, self.seed, save_file)
            self.load_save()

        elif self.mtype=="rotation":
            pso = Main.QuanEstimation.RotateBasis_Mopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                        self.decay_opt, self.gamma, self.povm_basis, self.W, self.eps)
            Main.QuanEstimation.CFIM_PSO_Mopt(pso, self.max_episode, self.particle_num, self.c0, self.c1, self.c2, self.seed, save_file)
            self.load_save()
        else:
            raise ValueError("{!r} is not a valid value for method, supported values are 'projection' and 'input'.".format(self.mtype))
        