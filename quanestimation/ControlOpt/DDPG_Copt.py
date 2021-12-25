import numpy as np
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control

class DDPG_Copt(Control.ControlSystem):
    def __init__(self, tspan, rho0, H0, Hc=[], dH=[], decay=[], ctrl_bound=[], W=[], \
                 ctrl0=[], max_episode=500, layer_num=3, layer_dim=200, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, accuracy=1e-8)

        """
        ----------
        Inputs
        ----------
        max_episode:
            --description: max number of the training episodes.
            --type: int

        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number of neurons in the hidden layer.
            --type: int
        
        seed:
            --description: random seed.
            --type: int

        """
        self.ctrl_interval = (len(self.tspan)-1)//len(self.control_coefficients[0])
        self.layer_num = layer_num
        self.layer_dim = layer_dim
        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the 
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and QFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """
        ddpg = Main.QuanEstimation.DDPG_Copt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, \
                self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.ctrl_interval, len(self.rho0), self.accuracy)
        Main.QuanEstimation.QFIM_DDPG_Copt(ddpg, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
    
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and CFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        ddpg = Main.QuanEstimation.DDPG_Copt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, \
                self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.ctrl_interval, len(self.rho0), self.accuracy)
        Main.QuanEstimation.CFIM_DDPG_Copt(Measurement, ddpg, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
