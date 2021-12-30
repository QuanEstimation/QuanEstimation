import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement

class DE_Mopt(Measurement.MeasurementSystem):
    def __init__(self, mtype, mgiven, tspan, rho0, H0, dH=[], decay=[], W=[], popsize=10, \
                measurement0=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        Measurement.MeasurementSystem.__init__(self, mtype, mgiven, tspan, rho0, H0, dH, decay, W, measurement0, seed, accuracy=1e-8)
        
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
        self.popsize =  popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        if self.mtype == 'projection':
            if measurement0 == []: 
                ini_measurement = [np.array(self.Measurement)]
            else:
                ini_measurement = measurement0
            self.ini_measurement = ini_measurement
        else: pass

        
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
        if self.mtype=='projection':
            diffevo = Main.QuanEstimation.projection_Mopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                    self.decay_opt, self.gamma, self.Measurement, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_DE_Mopt(diffevo, self.popsize, self.ini_measurement, self.c, self.cr, self.seed, self.max_episode, save_file)
            self.load_save()
        elif self.mtype=='sicpovm' or self.mtype=='given':
            diffevo = Main.QuanEstimation.givenpovm_Mopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                    self.decay_opt, self.gamma, self.povm_basis, self.M_num, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_DE_Mopt(diffevo, self.popsize, self.c, self.cr, self.seed, self.max_episode, save_file)
            self.load_save()
        else:
            raise ValueError("{!r} is not a valid value for mtype, supported values are 'projection', 'sicpovm' and 'given'.".format(self.mtype))
        