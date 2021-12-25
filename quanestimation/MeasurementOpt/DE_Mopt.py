import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement

class DE_Mopt(Measurement.MeasurementSystem):
    def __init__(self, tspan, rho0, H0, dH=[], decay=[], W=[], popsize=10, \
                measurement0=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        Measurement.MeasurementSystem.__init__(self, tspan, rho0, H0, dH, decay, W, measurement0, seed, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int
        
        measurement0:
           --description: initial guesses of measurements.
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
        if measurement0 == []: 
            ini_measurement = [np.array(self.Measurement)]
        else:
            ini_measurement = measurement0

        self.popsize =  popsize
        self.ini_measurement = ini_measurement
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
        Main.QuanEstimation.CFIM_DE_Mopt(diffevo, self.popsize, self.measurement, self.c, self.cr, self.seed, self.max_episode, save_file)
        self.load_save()
        