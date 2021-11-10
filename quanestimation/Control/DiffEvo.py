import numpy as np
from julia import Main
import quanestimation.Control.Control as Control
class DiffEvo(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                gamma=[], control_option=True, ctrl_bound=[-np.inf, np.inf], W=[], popsize=10, ini_population=[], \
                max_episodes=1000, c=1.0, cr=0.5, u0=0.1, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option, ctrl_bound, W)
        
        """
        --------
        inputs
        --------
        popsize:
           --description: number of particles.
           --type: int
        
        ini_population:
           --description: initial populations.
           --type: array

        max_episodes:
            --description: max number of training episodes.
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
            ini_population = [ctrl_initial]

        self.popsize =  popsize
        self.ini_population = ini_population
        self.max_episodes = max_episodes
        self.c = c
        self.cr = cr
        self.u0 = u0
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the 
                     QFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the nest episode and all the QFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """

        diffevo = Main.QuanEstimation.DiffEvo(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                  self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        Main.QuanEstimation.DE_QFIM(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.u0, self.seed, self.max_episodes, save_file)

        
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the 
                     CFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the nest episode and all the CFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """
        diffevo = Main.QuanEstimation.DiffEvo(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients)
        Main.QuanEstimation.DE_CFIM(Measurement, diffevo, self.popsize, self.ini_population, self.c, self.cr, self.u0, self.seed, self.max_episodes, save_file)
