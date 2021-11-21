from julia import Main
import quanestimation.StateOptimization.StateOptimization as stateopt
class StateOpt_AD(stateopt.StateOptSystem):
    def __init__(self, tspan, psi0, H0, dH=[], Decay=[], W=[], Adam=True, max_episode=300, \
                 lr=0.01, beta1=0.90, beta2=0.99, precision=1e-6):

        stateopt.StateOptSystem.__init__(self, tspan, psi0, H0, dH, Decay, W)

        """
        ----------
        Inputs
        ----------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        max_episode:
            --description: max number of training episodes.
            --type: int

        lr:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        precision:
            --description: calculation precision.
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0 
        self.precision = precision

    def QFIM(self, save_file=False):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the 
                     QFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the QFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """
        if self.gamma == [] or self.gamma[0] == 0.0:
            AD = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                 self.psi0, self.tspan, self.W)
            Main.QuanEstimation.AD_QFIM(AD, self.precision, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.max_episode, self.Adam, save_file)
        else:
            AD = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, \
                 self.psi0, self.tspan, self.Decay_opt, self.gamma, self.W)
            Main.QuanEstimation.AD_QFIM(AD, self.precision, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.max_episode, self.Adam, save_file)
        self.load_save()
            
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the 
                     CFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the CFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """

        if self.gamma == [] or self.gamma[0] == 0.0:
            AD = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                 self.psi0, self.tspan, self.W)
            Main.QuanEstimation.AD_CFIM(Measurement, AD, self.precision, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.max_episode, self.Adam, save_file)
        else:
            AD = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, \
                 self.psi0, self.tspan, self.Decay_opt, self.gamma, self.W)
            Main.QuanEstimation.AD_CFIM(Measurement, AD, self.precision, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.max_episode, self.Adam, save_file)
        self.load_save()