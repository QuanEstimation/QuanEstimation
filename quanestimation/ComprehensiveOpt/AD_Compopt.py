import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp


class AD_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        Adam=True,
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
    ):

        Comp.ComprehensiveSystem.__init__(
            self,
            psi0,
            ctrl0,
            measurement0,
            seed,
            eps=1e-8,
        )

        """
        ----------
        Inputs
        ----------

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

    def SC(self, target="QFIM", M=[], W=[], save_file=False):
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
        if self.dynamics_type != "dynamics":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is 'Lindblad dynamics'.".format(
                    self.dynamics_type
                )
            )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        AD = Main.QuanEstimation.Compopt_SCopt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.psi0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps,
        )
        if target == "QFIM":
            Main.QuanEstimation.SC_AD_Compopt(
                AD,
                self.max_episode,
                self.epsilon,
                self.mt,
                self.vt,
                self.beta1,
                self.beta2,
                self.eps,
                self.Adam,
                save_file,
            )
            self.load_save_state()
        elif target == "CFIM":
            warnings.warn(
                "AD is not available when target='CFIM'. Supported methods are 'PSO' and 'DE'.",
                DeprecationWarning,
            )
