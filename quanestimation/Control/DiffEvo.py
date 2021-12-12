import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class DiffEvo(Control.ControlSystem):
    def __init__(self, tspan, rho0, H0, Hc=[], dH=[], ctrl_0=[], decay=[], ctrl_bound=[], W=[], \
                popsize=10, ini_population=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho0, H0, Hc, dH, ctrl_0, decay, ctrl_bound, W, accuracy=1e-8)
        
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
            ini_population = [np.array(self.control_coefficients)]

        self.popsize =  popsize
        self.ini_population = ini_population
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the 
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and 
                                 all the QFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """

        diffevo = Main.QuanEstimation.DiffEvo(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan, self.decay_opt, \
                            self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W, self.accuracy)
        Main.QuanEstimation.DE_QFIM(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, self.max_episode, save_file)

        
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        diffevo = Main.QuanEstimation.DiffEvo(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan, self.decay_opt, \
                            self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W, self.accuracy)
        Main.QuanEstimation.DE_CFIM(Measurement, diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, self.max_episode, save_file)
