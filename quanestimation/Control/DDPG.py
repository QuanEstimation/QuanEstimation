import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class DDPG(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], control_option=True, ctrl_bound=[-np.inf, np.inf], W=[], layer_num=3, layer_dim=200, \
                 max_episodes=500, seed=1234):

        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option, ctrl_bound, W)

        """
        ----------
        Inputs
        ----------
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
        self.ctrl_interval = len(self.tspan)//len(self.control_coefficients[0])
        self.layer_num = layer_num
        self.layer_dim = layer_dim
        self.max_episodes = max_episodes
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the 
                     QFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and QFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """
        params = Main.QuanEstimation.ControlEnvParams(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, \
                self.tspan, self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.ctrl_interval, len(self.rho_initial))
        Main.QuanEstimation.DDPG_QFIM(params, self.layer_num, self.layer_dim, self.seed, self.max_episodes, save_file)
    
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the 
                     CFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and CFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """
        params = Main.QuanEstimation.ControlEnvParams(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, \
                self.tspan, self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.ctrl_interval, len(self.rho_initial))
        Main.QuanEstimation.DDPG_CFIM(Measurement, params, self.layer_num, self.layer_dim, self.seed, self.max_episodes, save_file)
