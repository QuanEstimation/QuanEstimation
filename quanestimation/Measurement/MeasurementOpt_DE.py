import numpy as np
from julia import Main
import quanestimation.Measurement.Measurement_struct as Measurement_struct

class DiffEvo_Meas(Measurement_struct.MeasurementSystem):
    def __init__(self, tspan, rho0, H0, dH=[], decay=[], W=[], ini_measurement=[], \
                popsize=10, ini_population=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        Measurement_struct.MeasurementSystem.__init__(self, tspan, rho0, H0, dH, decay, W, ini_measurement, seed, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int
        
        ini_population:
           --description: initial populations.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        c:
            --description: mutation constant.
            --type: float

        cr:
            --description: crossover constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """
        if ini_population == []: 
            ini_population = [np.array(self.Measurement)]

        self.popsize =  popsize
        self.ini_population = ini_population
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed
        
    def CFIM(self, save_file=False):
        """
        Description: use differential evolution algorithm to update the measurements that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the measurements for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the measurements for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        diffevo = Main.QuanEstimation.MeasurementOpt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                    self.decay_opt, self.gamma, self.Measurement, self.W, self.accuracy)
        Main.QuanEstimation.DE_CFIM(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, self.max_episode, save_file)
        self.load_save()
        