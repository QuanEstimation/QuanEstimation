from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State


class AD_Sopt(State.StateSystem):
    def __init__(
        self,
        psi0=[],
        Adam=False,
        max_episode=300,
        seed=1234,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        load=False,
    ):

        State.StateSystem.__init__(self, psi0, seed=seed, load=load, eps=1e-8)

        """
        ----------
        Inputs
        ----------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

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

        eps:
            --description: calculation eps.
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0

    def QFIM(self, W=[], save_file=False):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the QFI (Tr(WF^{-1})).
                           False: save the initial states for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if any(self.gamma):
                AD = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.QFIM_AD_Sopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    save_file,
                )
            else:
                AD = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.QFIM_AD_Sopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    save_file,
                )
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            AD = Main.QuanEstimation.TimeIndepend_Kraus(
                self.K, self.dK, self.psi0, self.W, self.eps
            )
            Main.QuanEstimation.QFIM_AD_Sopt(
                AD,
                self.mt,
                self.vt,
                self.epsilon,
                self.beta1,
                self.beta2,
                self.max_episode,
                self.Adam,
                save_file,
            )

        self.load_save()

    def CFIM(self, M, W=[], save_file=False):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the initial states for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if any(self.gamma):
                AD = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_AD_Sopt(
                    M,
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    save_file,
                )
            else:
                AD = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_AD_Sopt(
                    M,
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    save_file,
                )
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            AD = Main.QuanEstimation.TimeIndepend_Kraus(
                self.K, self.dK, self.psi0, self.W, self.eps
            )
            Main.QuanEstimation.CFIM_AD_Sopt(
                AD,
                self.mt,
                self.vt,
                self.epsilon,
                self.beta1,
                self.beta2,
                self.max_episode,
                self.Adam,
                save_file,
            )

        self.load_save()

    def HCRB(self, W=[], save_file=False):
        warnings.warn(
            "AD is not available when the objective function is HCRB. Supported methods are 'PSO', 'DE', 'NM' and 'DDPG'.",
            DeprecationWarning,
        )
