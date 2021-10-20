import numpy as np
from julia import Main
import quanestimation.Control.Control as Control

class NelderMead(Control.ControlSystem):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
                gamma=[], control_option=True, ctrl_bound=1.0, W=[], state_num=10, ini_state=[], coeff_r=1.0, coeff_e=2.0, \
                coeff_c=0.5, coeff_s=0.5, seed=1234, max_episodes=200, epsilon=1e-3):

        Control.ControlSystem.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                                       gamma, control_option)
        self.state_num = state_num
        self.ini_state = ini_state
        self.ctrl_bound = ctrl_bound
        self.coeff_r = coeff_r
        self.coeff_e = coeff_e
        self.coeff_c = coeff_c
        self.coeff_s = coeff_s
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.seed = seed
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

    def QFIM(self, save_file=False):
        neldermead = Main.QuanEstimation.NelderMead(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        if len(self.Hamiltonian_derivative) == 1:
            Main.QuanEstimation.NelderMead_QFI(neldermead, self.state_num, self.ini_state, self.coeff_r, self.coeff_e, self.coeff_c, self.coeff_s, self.epsilon, self.max_episodes, self.seed, save_file)
        else:
            Main.QuanEstimation.NelderMead_QFIM(neldermead, self.state_num, self.ini_state, self.coeff_r, self.coeff_e, self.coeff_c, self.coeff_s, self.epsilon, self.max_episodes, self.seed, save_file)

    def CFIM(self, M, save_file=False):
        neldermead = Main.QuanEstimation.NelderMead(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound, self.W)
        if len(self.Hamiltonian_derivative) == 1:
            Main.QuanEstimation.NelderMead_CFI(M, neldermead, self.state_num, self.ini_state, self.coeff_r, self.coeff_e, self.coeff_c, self.coeff_s, self.epsilon, self.max_episodes, self.seed, save_file)
        else:
            Main.QuanEstimation.NelderMead_CFIM(M, neldermead, self.state_num, self.ini_state, self.coeff_r, self.coeff_e, self.coeff_c, self.coeff_s, self.epsilon, self.max_episodes, self.seed, save_file)

            