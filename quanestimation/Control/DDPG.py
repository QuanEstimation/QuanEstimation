import numpy as np
# import warnings
from julia import Main
import quanestimation.Control.Control as Control

class DDPG(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                 gamma=[], control_option=True, ctrl_bound=[-np.inf, np.inf], W=[], seed=1234, layer_num=3, layer_dim=200):

        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option, ctrl_bound, W)

        """
        ----------
        Inputs
        ----------
        """
        self.ctrl_interval = len(tspan)//len(ctrl_initial[0])
        self.seed = seed
        self.layer_num = layer_num
        self.layer_dim = layer_dim

    def QFIM(self, save_file=False):
        """
        """
        params = Main.QuanEstimation.ControlEnvParams(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, \
                self.tspan, self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_interval, self.ctrl_bound, self.W, len(self.rho_initial))
        # env = Main.QuanEstimation.ControlEnv(params=params)
        
        if save_file == False:
            Main.QuanEstimation.DDPG_QFIM(params, self.seed, self.layer_num, self.layer_dim)
    