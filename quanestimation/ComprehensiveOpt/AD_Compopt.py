import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp

class AD_Compopt(Comp.ComprehensiveSystem):
    def __init__(self, option, tspan, H0, dH, Hc, decay=[], ctrl_bound=[], W=[], psi0=[], measurement0=[],\
                Adam=True, ctrl0=[], max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, seed=1234):

        Comp.ComprehensiveSystem.__init__(self, tspan, psi0, measurement0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, seed, accuracy=1e-8)

        """
        ----------
        Inputs
        ----------
        auto:
            --description: True: use autodifferential to calculate the gradient.
                                  False: calculate the gradient with analytical method.
            --type: bool (True or False)

        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        ctrl0:
           --description: initial guess of controls.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

        epsilon:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        """
        
        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed

    def SC(self, target="QFIM", M=[], save_file=False):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to optimize states and control coefficients.

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the states and control coefficients and the value of target function.
                           False: save the states and control coefficients for the last episode and all the value of target function.
            --type: bool

        """
        AD = Main.QuanEstimation.Compopt_SCopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.accuracy)
        if target == "QFIM":
            Main.QuanEstimation.SC_AD_Compopt(AD, self.max_episode, self.epsilon, self.mt, self.vt, self.beta1, self.beta2, self.accuracy, self.Adam, save_file)
            self.load_save_state()
        elif target == "CFIM":
            warnings.warn("AD is not available when target='CFIM'. Supported methods are 'PSO' and 'DE'.", DeprecationWarning)
