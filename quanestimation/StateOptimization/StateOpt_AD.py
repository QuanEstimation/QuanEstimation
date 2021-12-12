from julia import Main
import quanestimation.StateOptimization.StateOptimization as stateopt
class StateOpt_AD(stateopt.StateOptSystem):
    def __init__(self, tspan, psi0, H0, dH=[], decay=[], W=[], Adam=True, max_episode=300, \
                 epsilon=0.01, beta1=0.90, beta2=0.99):

        stateopt.StateOptSystem.__init__(self, tspan, psi0, H0, dH, decay, W, accuracy=1e-8)

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

        accuracy:
            --description: calculation accuracy.
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0 

    def QFIM(self, save_file=False):
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
        if any(self.gamma):
            AD = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                        self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.AD_QFIM(AD, self.mt, self.vt, self.epsilon, self.beta1, self.beta2, self.max_episode, \
                                        self.Adam, save_file)
        else:
            AD = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                            self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.AD_QFIM(AD, self.mt, self.vt, self.epsilon, self.beta1, self.beta2, self.max_episode, \
                                        self.Adam, save_file)
        self.load_save()
            
    def CFIM(self, Measurement, save_file=False):
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

        if any(self.gamma):
            AD = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                        self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.AD_CFIM(Measurement, AD, self.mt, self.vt, self.epsilon, self.beta1, self.beta2, \
                                        self.max_episode, self.Adam, save_file)
        else:
            AD = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                            self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.AD_CFIM(Measurement, AD, self.mt, self.vt, self.epsilon, self.beta1, self.beta2, \
                                        self.max_episode, self.Adam, save_file)
        self.load_save()